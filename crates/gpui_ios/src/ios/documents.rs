//! The iPadOS document picker: how files cross the app-sandbox boundary.
//!
//! An iPadOS app can only read and write inside its own container. Everything else the user owns
//! (iCloud Drive, Downloads, "On My iPad", third-party providers) is reached through
//! `UIDocumentPickerViewController`, and `std::fs` on a path outside the container fails. Both
//! directions therefore go through a copy:
//!
//! * [`export_files`] — the app first writes the file into its own container (its temporary
//!   directory is the natural place), then presents the picker in export mode; the user chooses a
//!   destination folder and the system copies the file there. There is no "save panel" that hands
//!   back a path to write to, so unlike desktop platforms the file must exist before the user
//!   decides where it goes.
//! * [`pick_files`] — the picker returns *local copies* of the chosen files (in the app's
//!   temporary inbox), so the caller reads them with ordinary `std::fs` and never has to manage
//!   security-scoped access.
//!
//! The picker is presented modally from the topmost view controller; a second request while one is
//! showing fails rather than stacking.

use anyhow::{Result, anyhow};
use futures::channel::oneshot;
use objc2::{
    DefinedClass, MainThreadMarker, MainThreadOnly, define_class, msg_send,
    rc::Retained,
    runtime::ProtocolObject,
};
use objc2_foundation::{NSArray, NSObject, NSObjectProtocol, NSString, NSURL};
use objc2_ui_kit::{UIDocumentPickerDelegate, UIDocumentPickerViewController};
use objc2_uniform_type_identifiers::UTType;
use std::{cell::RefCell, path::PathBuf};

use super::IosPlatform;

/// `Ok(None)`: the user cancelled. `Ok(Some(paths))`: the exported destinations / the imported
/// local copies.
pub type PickerOutcome = Result<Option<Vec<PathBuf>>>;

struct DelegateIvars {
    sender: RefCell<Option<oneshot::Sender<PickerOutcome>>>,
}

define_class!(
    #[unsafe(super = NSObject)]
    #[thread_kind = MainThreadOnly]
    #[name = "GPUIDocumentPickerDelegate"]
    #[ivars = DelegateIvars]
    struct PickerDelegate;

    unsafe impl NSObjectProtocol for PickerDelegate {}

    unsafe impl UIDocumentPickerDelegate for PickerDelegate {
        #[unsafe(method(documentPicker:didPickDocumentsAtURLs:))]
        fn did_pick_documents(
            &self,
            _controller: &UIDocumentPickerViewController,
            urls: &NSArray<NSURL>,
        ) {
            let paths = urls
                .iter()
                .filter_map(|url| url.path().map(|path| PathBuf::from(path.to_string())))
                .collect();
            self.finish(Ok(Some(paths)));
        }

        #[unsafe(method(documentPickerWasCancelled:))]
        fn was_cancelled(&self, _controller: &UIDocumentPickerViewController) {
            self.finish(Ok(None));
        }
    }
);

impl PickerDelegate {
    fn new(sender: oneshot::Sender<PickerOutcome>, mtm: MainThreadMarker) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(DelegateIvars {
            sender: RefCell::new(Some(sender)),
        });
        unsafe { msg_send![super(this), init] }
    }

    fn finish(&self, outcome: PickerOutcome) {
        if let Some(sender) = self.ivars().sender.borrow_mut().take() {
            if sender.send(outcome).is_err() {
                log::debug!("GPUI iOS: document picker receiver was dropped");
            }
        }
        // `UIDocumentPickerViewController.delegate` does not retain its delegate.
        ACTIVE.with_borrow_mut(|active| *active = None);
    }
}

thread_local! {
    static ACTIVE: RefCell<Option<Retained<PickerDelegate>>> = const { RefCell::new(None) };
}

fn present(
    build: impl FnOnce(MainThreadMarker) -> Retained<UIDocumentPickerViewController>,
) -> oneshot::Receiver<PickerOutcome> {
    let (sender, receiver) = oneshot::channel();
    let fail = |sender: oneshot::Sender<PickerOutcome>, message: &str| {
        if sender.send(Err(anyhow!("{message}"))).is_err() {
            log::debug!("GPUI iOS: document picker receiver was dropped");
        }
    };
    let Some(mtm) = MainThreadMarker::new() else {
        fail(sender, "the document picker must be presented from the main thread");
        return receiver;
    };
    if ACTIVE.with_borrow(Option::is_some) {
        fail(sender, "a document picker is already showing");
        return receiver;
    }
    let Some(presenter) = IosPlatform::presented_view_controller() else {
        fail(sender, "there is no view controller to present the document picker from");
        return receiver;
    };

    let delegate = PickerDelegate::new(sender, mtm);
    let picker = build(mtm);
    picker.setDelegate(Some(ProtocolObject::from_ref(&*delegate)));
    picker.setShouldShowFileExtensions(true);
    ACTIVE.with_borrow_mut(|active| *active = Some(delegate));
    presenter.presentViewController_animated_completion(&picker, true, None);
    receiver
}

/// Let the user copy `paths` (files inside the app's container) to a folder of their choice.
/// The result holds the destinations the system created; the caller still owns the originals.
pub fn export_files(paths: &[PathBuf]) -> oneshot::Receiver<PickerOutcome> {
    present(|mtm| {
        let urls: Vec<Retained<NSURL>> = paths
            .iter()
            .map(|path| NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy())))
            .collect();
        UIDocumentPickerViewController::initForExportingURLs_asCopy(
            UIDocumentPickerViewController::alloc(mtm),
            &NSArray::from_retained_slice(&urls),
            true,
        )
    })
}

/// Let the user choose files to open. `extensions` (without the dot) restricts what can be picked;
/// empty allows any file. The result holds local copies inside the app's container.
pub fn pick_files(extensions: &[&str], multiple: bool) -> oneshot::Receiver<PickerOutcome> {
    present(|mtm| {
        let mut types: Vec<Retained<UTType>> = extensions
            .iter()
            .filter_map(|extension| UTType::typeWithFilenameExtension(&NSString::from_str(extension)))
            .collect();
        if types.is_empty() {
            types.extend(UTType::typeWithIdentifier(&NSString::from_str("public.data")));
        }
        let picker = UIDocumentPickerViewController::initForOpeningContentTypes_asCopy(
            UIDocumentPickerViewController::alloc(mtm),
            &NSArray::from_retained_slice(&types),
            true,
        );
        picker.setAllowsMultipleSelection(multiple);
        picker
    })
}

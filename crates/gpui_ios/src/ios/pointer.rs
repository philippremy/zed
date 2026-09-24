//! The cursor shape of an attached trackpad or mouse.
//!
//! iPadOS draws the pointer itself; an app can only ask it to morph via a `UIPointerInteraction`.
//! One interaction covers the whole window view, and GPUI's cursor style is mapped onto the two
//! shapes iPadOS lets an app request over plain content: the text beam and the system pointer.

use gpui::CursorStyle;
use objc2::{
    DefinedClass, MainThreadMarker, MainThreadOnly, define_class, msg_send, rc::Retained,
    runtime::ProtocolObject,
};
use objc2_foundation::{NSObject, NSObjectProtocol};
use objc2_ui_kit::{
    UIAxis, UIPointerInteraction, UIPointerInteractionDelegate, UIPointerRegion, UIPointerShape,
    UIPointerStyle, UIView,
};
use std::cell::{Cell, RefCell};

const BEAM_LENGTH: f64 = 20.;

struct DelegateIvars {
    style: Cell<CursorStyle>,
}

define_class!(
    #[unsafe(super = NSObject)]
    #[thread_kind = MainThreadOnly]
    #[name = "GPUIPointerDelegate"]
    #[ivars = DelegateIvars]
    struct PointerDelegate;

    unsafe impl NSObjectProtocol for PointerDelegate {}

    unsafe impl UIPointerInteractionDelegate for PointerDelegate {
        #[unsafe(method_id(pointerInteraction:styleForRegion:))]
        fn style_for_region(
            &self,
            _interaction: &UIPointerInteraction,
            _region: &UIPointerRegion,
        ) -> Option<Retained<UIPointerStyle>> {
            let mtm = self.mtm();
            let beam = |axis| {
                UIPointerStyle::styleWithShape_constrainedAxes(
                    &UIPointerShape::beamWithPreferredLength_axis(BEAM_LENGTH, axis, mtm),
                    UIAxis::empty(),
                )
            };
            Some(match self.ivars().style.get() {
                CursorStyle::IBeam => beam(UIAxis::Vertical),
                CursorStyle::IBeamCursorForVerticalLayout => beam(UIAxis::Horizontal),
                _ => UIPointerStyle::systemPointerStyle(mtm),
            })
        }
    }
);

impl PointerDelegate {
    fn new(mtm: MainThreadMarker) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(DelegateIvars {
            style: Cell::new(CursorStyle::Arrow),
        });
        unsafe { msg_send![super(this), init] }
    }
}

struct Pointer {
    interaction: Retained<UIPointerInteraction>,
    delegate: Retained<PointerDelegate>,
}

thread_local! {
    // The interaction does not retain its delegate.
    static POINTER: RefCell<Option<Pointer>> = const { RefCell::new(None) };
}

/// Attach the pointer interaction to `view` (the first window's view; later windows are ignored).
pub(super) fn install(view: &UIView, mtm: MainThreadMarker) {
    POINTER.with_borrow_mut(|pointer| {
        if pointer.is_some() {
            return;
        }
        let delegate = PointerDelegate::new(mtm);
        let interaction = UIPointerInteraction::initWithDelegate(
            UIPointerInteraction::alloc(mtm),
            Some(ProtocolObject::from_ref(&*delegate)),
        );
        view.addInteraction(ProtocolObject::from_ref(&*interaction));
        *pointer = Some(Pointer {
            interaction,
            delegate,
        });
    });
}

pub(super) fn set_cursor_style(style: CursorStyle) {
    POINTER.with_borrow(|pointer| {
        let Some(pointer) = pointer else { return };
        if pointer.delegate.ivars().style.replace(style) != style {
            pointer.interaction.invalidate();
        }
    });
}

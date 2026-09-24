//! iOS Window implementation using UIWindow and UIViewController.
//!
//! iOS windows are fundamentally different from desktop windows:
//! - Always fullscreen (or split-screen on iPad)
//! - No title bar or window chrome
//! - Touch-based input
//! - Safe area insets for notch/home indicator
//!
//! The window is backed by a UIWindow containing a UIViewController
//! whose view hosts a CAMetalLayer.

use super::IosDisplay;
use super::events::*;
use super::text_input::TextInputView;
use gpui::{
    AnyWindowHandle, Bounds, Capslock, DevicePixels, DispatchEventResult, Edges, EditMenuActions,
    GpuSpecs, Modifiers, Pixels, PlatformAtlas, PlatformDisplay, PlatformInput,
    PlatformInputHandler, PlatformWindow, Point, PromptButton, PromptLevel, RequestFrameOptions,
    MouseExitEvent, MouseMoveEvent, PinchEvent, ScrollDelta, ScrollWheelEvent, Scene, Size,
    TextInputConfiguration, TextInputStateChange, TouchEvent, TouchId, TouchPhase,
    WindowAppearance, WindowBackgroundAppearance, WindowBounds, WindowControlArea, WindowInsets,
    WindowParams, WindowVisibility, px, size,
};
use gpui_apple::metal_renderer::{Context as MetalContext, MetalRenderer};
use objc2::rc::Retained;
use objc2::runtime::{AnyClass, AnyObject, Sel};
use objc2::{
    ClassType, DefinedClass, MainThreadMarker, MainThreadOnly, define_class, msg_send, sel,
};
use objc2_core_foundation::{CGPoint, CGRect, CGSize};
use objc2_foundation::{NSString, NSArray, NSNotification, NSNotificationCenter, NSObjectProtocol, NSSet, NSValue};
use objc2_quartz_core::CAMetalLayer;
use objc2_ui_kit::{
    NSValueUIGeometryExtensions, UIKeyboardFrameEndUserInfoKey,
    UIKeyboardWillChangeFrameNotification, UIKeyboardWillHideNotification,
    UIResponderStandardEditActions, UIStatusBarStyle,
};
use objc2_ui_kit::{
    UICoordinateSpace, UIEdgeInsets, UIEditMenuConfiguration, UIEditMenuInteraction, UIEvent,
    UIAlertAction, UIAlertActionStyle, UIAlertController, UIAlertControllerStyle,
    UIGestureRecognizerState, UIHoverGestureRecognizer, UIPanGestureRecognizer,
    UIPinchGestureRecognizer, UIScreen, UIScrollTypeMask, UITouch,
    UITraitCollection, UITraitEnvironment, UIUserInterfaceStyle, UIView, UIViewAutoresizing,
    UIViewController, UIViewLayoutRegion, UIViewLayoutRegionAdaptivityAxis, UIWindow,
    UIWindowScene,
};
use parking_lot::Mutex;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, UiKitDisplayHandle, UiKitWindowHandle};
use std::{
    cell::{Cell, RefCell},
    ffi::c_void,
    ptr::NonNull,
    rc::{Rc, Weak},
    sync::Arc,
};

const KEYBOARD_DISMISS_DISTANCE: Pixels = px(24.);

#[derive(Clone, Copy)]
struct KeyboardDismissTouch {
    id: TouchId,
    start_position: Point<Pixels>,
}

static KEYBOARD_OBSERVERS_REGISTERED: std::sync::Once = std::sync::Once::new();

/// Global storage for the current status bar style.
/// 0 = default (dark content), 1 = light content.
/// Accessed from the main thread only.
static STATUS_BAR_STYLE: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

#[derive(Default)]
pub(super) struct WindowReference(pub(super) RefCell<Weak<IosWindowState>>);

impl WindowReference {
    pub(super) fn with_window<R>(&self, callback: impl FnOnce(&IosWindowState) -> R) -> Option<R> {
        let window = self.0.borrow().upgrade()?;
        // Keep callback storage and native objects alive even if the callback
        // synchronously closes the platform window.
        Some(callback(&window))
    }

    pub(super) fn dispatch_edit_menu_shortcut(&self, key: &str) {
        self.with_window(|window| window.dispatch_edit_menu_shortcut(key));
    }

    pub(super) fn can_perform_action(&self, action: Sel) -> bool {
        self.with_window(|window| {
            let actions = window.edit_menu_actions.get();
            (action == sel!(cut:) && actions.cut)
                || (action == sel!(copy:) && actions.copy)
                || (action == sel!(paste:) && actions.paste)
                || (action == sel!(selectAll:) && actions.select_all)
                || window.can_edit_from_menu_bar(action)
        })
        .unwrap_or(false)
    }
}

struct CallbackSlot<T> {
    value: RefCell<Option<T>>,
    generation: Cell<u64>,
}

impl<T> Default for CallbackSlot<T> {
    fn default() -> Self {
        Self {
            value: RefCell::new(None),
            generation: Cell::new(0),
        }
    }
}

impl<T> CallbackSlot<T> {
    fn set(&self, value: T) {
        self.generation.set(self.generation.get().wrapping_add(1));
        drop(self.value.replace(Some(value)));
    }

    fn take(&self) -> Option<T> {
        // Even an empty take can mean the focused handler was withdrawn during a callback.
        self.generation.set(self.generation.get().wrapping_add(1));
        self.value.borrow_mut().take()
    }

    fn with<R>(&self, callback: impl FnOnce(&mut T) -> R) -> Option<R> {
        let mut value = self.value.borrow_mut().take()?;
        let generation = self.generation.get();
        let result = callback(&mut value);
        // Do not resurrect a handler that application code replaced or removed.
        if self.generation.get() == generation {
            drop(self.value.replace(Some(value)));
        }
        Some(result)
    }
}

define_class!(
    #[unsafe(super = UIViewController)]
    #[thread_kind = MainThreadOnly]
    #[name = "GPUIViewController"]
    #[ivars = WindowReference]
    struct ViewController;

    impl ViewController {
        #[unsafe(method(preferredStatusBarStyle))]
        fn preferred_status_bar_style(&self) -> UIStatusBarStyle {
            let style = STATUS_BAR_STYLE.load(std::sync::atomic::Ordering::Relaxed);
            if style == 1 {
                UIStatusBarStyle::LightContent
            } else {
                UIStatusBarStyle::DarkContent
            }
        }

        #[unsafe(method(viewDidLayoutSubviews))]
        fn view_did_layout_subviews(&self) {
            unsafe {
                let _: () = msg_send![super(self), viewDidLayoutSubviews];
            }
            self.ivars().with_window(IosWindowState::handle_layout_change);
        }
    }
);

impl ViewController {
    fn new(main_thread: MainThreadMarker) -> Retained<Self> {
        let this = Self::alloc(main_thread).set_ivars(WindowReference::default());
        unsafe { msg_send![super(this), init] }
    }
}

/// Set the iOS status bar content style (light or dark text/icons).
///
/// This updates the stored style and asks the root view controller
/// to re-query `preferredStatusBarStyle`.
pub fn set_status_bar_style(style: crate::StatusBarContentStyle) {
    use crate::StatusBarContentStyle;

    let value = match style {
        StatusBarContentStyle::Light => 1,
        StatusBarContentStyle::Dark => 0,
    };
    STATUS_BAR_STYLE.store(value, std::sync::atomic::Ordering::Relaxed);

    super::application::with_windows(|window| {
        window.view_controller.setNeedsStatusBarAppearanceUpdate();
    });
}

define_class!(
    #[unsafe(super = UIView)]
    #[thread_kind = MainThreadOnly]
    #[name = "GPUIMetalView"]
    #[ivars = WindowReference]
    struct MetalView;

    unsafe impl NSObjectProtocol for MetalView {}

    impl MetalView {
        #[unsafe(method(layerClass))]
        fn layer_class() -> &'static AnyClass {
            CAMetalLayer::class()
        }

        // This callback also covers iOS 15/16, before UIKit's trait-registration API.
        #[unsafe(method(traitCollectionDidChange:))]
        fn trait_collection_did_change(&self, previous: Option<&UITraitCollection>) {
            unsafe {
                let _: () = msg_send![super(self), traitCollectionDidChange: previous];
                if self
                    .traitCollection()
                    .hasDifferentColorAppearanceComparedToTraitCollection(previous)
                {
                    self.ivars()
                        .with_window(IosWindowState::notify_appearance_changed);
                }
            }
        }

        #[unsafe(method(handlePinch:))]
        fn handle_pinch(&self, recognizer: &UIPinchGestureRecognizer) {
            let position = recognizer_location(recognizer, self);
            // The scale is cumulative from the gesture's start; resetting it after every event
            // makes each one carry only its own change, which is what `PinchEvent::delta` is.
            let delta = recognizer.scale() as f32 - 1.;
            recognizer.setScale(1.);
            let phase = gesture_phase(recognizer.state());
            self.ivars()
                .with_window(|window| window.dispatch_pinch(position, delta, phase));
        }

        #[unsafe(method(handleHover:))]
        fn handle_hover(&self, recognizer: &UIHoverGestureRecognizer) {
            let position = recognizer_location(recognizer, self);
            self.ivars()
                .with_window(|window| window.dispatch_hover(position, recognizer.state()));
        }

        #[unsafe(method(handleScroll:))]
        fn handle_scroll(&self, recognizer: &UIPanGestureRecognizer) {
            let position = recognizer_location(recognizer, self);
            let translation = recognizer.translationInView(Some(self));
            recognizer.setTranslation_inView(CGPoint::ZERO, Some(self));
            let delta = Point::new(px(translation.x as f32), px(translation.y as f32));
            let phase = gesture_phase(recognizer.state());
            self.ivars()
                .with_window(|window| window.dispatch_pointer_scroll(position, delta, phase));
        }

        #[unsafe(method(touchesBegan:withEvent:))]
        fn touches_began(&self, touches: &NSSet<UITouch>, event: Option<&UIEvent>) {
            self.handle_touches(touches, event);
        }

        #[unsafe(method(touchesMoved:withEvent:))]
        fn touches_moved(&self, touches: &NSSet<UITouch>, event: Option<&UIEvent>) {
            self.handle_touches(touches, event);
        }

        #[unsafe(method(touchesEnded:withEvent:))]
        fn touches_ended(&self, touches: &NSSet<UITouch>, event: Option<&UIEvent>) {
            self.handle_touches(touches, event);
        }

        #[unsafe(method(touchesCancelled:withEvent:))]
        fn touches_cancelled(&self, touches: &NSSet<UITouch>, event: Option<&UIEvent>) {
            self.handle_touches(touches, event);
        }

        #[unsafe(method(canPerformAction:withSender:))]
        fn can_perform_action(&self, action: Sel, _sender: Option<&AnyObject>) -> bool {
            self.ivars().can_perform_action(action)
        }
    }

    unsafe impl UIResponderStandardEditActions for MetalView {
        #[unsafe(method(cut:))]
        unsafe fn cut(&self, _sender: Option<&AnyObject>) {
            self.ivars().dispatch_edit_menu_shortcut("x");
        }

        #[unsafe(method(copy:))]
        unsafe fn copy(&self, _sender: Option<&AnyObject>) {
            self.ivars().dispatch_edit_menu_shortcut("c");
        }

        #[unsafe(method(paste:))]
        unsafe fn paste(&self, _sender: Option<&AnyObject>) {
            self.ivars().dispatch_edit_menu_shortcut("v");
        }

        #[unsafe(method(selectAll:))]
        unsafe fn select_all(&self, _sender: Option<&AnyObject>) {
            self.ivars().dispatch_edit_menu_shortcut("a");
        }
    }
);

impl MetalView {
    fn new(frame: CGRect, main_thread: MainThreadMarker) -> Retained<Self> {
        let this = Self::alloc(main_thread).set_ivars(WindowReference::default());
        unsafe { msg_send![super(this), initWithFrame: frame] }
    }

    /// Pointer input that UIKit does not report as touches: trackpad/mouse hover, two-finger
    /// scrolling and the pinch gesture (from a trackpad or two fingers).
    fn install_gesture_recognizers(&self) {
        // SAFETY: the target is this view, which the gesture recognizers' actions are defined on.
        unsafe {
            let pinch = UIPinchGestureRecognizer::initWithTarget_action(
                UIPinchGestureRecognizer::alloc(self.mtm()),
                Some(self),
                Some(sel!(handlePinch:)),
            );
            // Recognising a pinch cancels the two touches, so they never end up as a stray tap.
            pinch.setCancelsTouchesInView(true);
            self.addGestureRecognizer(&pinch);

            let hover = UIHoverGestureRecognizer::initWithTarget_action(
                UIHoverGestureRecognizer::alloc(self.mtm()),
                Some(self),
                Some(sel!(handleHover:)),
            );
            self.addGestureRecognizer(&hover);

            let scroll = UIPanGestureRecognizer::initWithTarget_action(
                UIPanGestureRecognizer::alloc(self.mtm()),
                Some(self),
                Some(sel!(handleScroll:)),
            );
            // Scroll events only: finger drags already arrive as touches and are recognised by GPUI.
            scroll.setAllowedTouchTypes(&NSArray::new());
            scroll.setAllowedScrollTypesMask(UIScrollTypeMask::All);
            self.addGestureRecognizer(&scroll);
            }
    }

    fn handle_touches(&self, touches: &NSSet<UITouch>, event: Option<&UIEvent>) {
        for touch in touches {
            self.ivars()
                .with_window(|window| window.handle_touch(&touch, event));
        }
    }
}

pub(crate) struct IosWindow {
    state: Rc<IosWindowState>,
}

impl std::ops::Deref for IosWindow {
    type Target = IosWindowState;

    fn deref(&self) -> &Self::Target {
        &self.state
    }
}

#[allow(clippy::type_complexity)]
pub(crate) struct IosWindowState {
    /// GPUI's handle for this window, reported by `Platform::active_window`
    handle: AnyWindowHandle,
    /// Whether the scene is currently active
    active: Cell<bool>,
    /// The UIWindow object
    window: Retained<UIWindow>,
    /// The UIViewController
    view_controller: Retained<ViewController>,
    /// The Metal-backed UIView
    view: Retained<MetalView>,
    /// The hidden text input view for keyboard input
    text_input_view: Retained<TextInputView>,
    edit_menu_interaction: Option<Retained<UIEditMenuInteraction>>,
    edit_menu_actions: Cell<EditMenuActions>,
    /// Current bounds in pixels
    bounds: Cell<Bounds<Pixels>>,
    /// Scale factor
    scale_factor: Cell<f32>,
    /// Input handler for text input
    input_handler: CallbackSlot<PlatformInputHandler>,
    request_frame_callback: CallbackSlot<Box<dyn FnMut(RequestFrameOptions)>>,
    force_next_frame: Cell<bool>,
    /// Callback for input events
    input_callback: CallbackSlot<Box<dyn FnMut(PlatformInput) -> DispatchEventResult>>,
    /// Callback for active status changes
    active_status_callback: CallbackSlot<Box<dyn FnMut(bool)>>,
    visibility: Cell<WindowVisibility>,
    visibility_callback: CallbackSlot<Box<dyn FnMut(WindowVisibility)>>,
    /// Callback for hover status changes (not really applicable on iOS)
    hover_status_callback: CallbackSlot<Box<dyn FnMut(bool)>>,
    /// Callback for resize events
    resize_callback: CallbackSlot<Box<dyn FnMut(Size<Pixels>, f32)>>,
    /// Callback for move events (not applicable on iOS)
    moved_callback: CallbackSlot<Box<dyn FnMut()>>,
    /// Callback for should close
    should_close_callback: CallbackSlot<Box<dyn FnMut() -> bool>>,
    /// Callback for hit test
    hit_test_callback: CallbackSlot<Box<dyn FnMut() -> Option<WindowControlArea>>>,
    /// Callback for close
    close_callback: CallbackSlot<Box<dyn FnOnce()>>,
    /// Callback for appearance changes
    appearance_changed_callback: CallbackSlot<Box<dyn FnMut()>>,
    insets_changed_callback: CallbackSlot<Box<dyn FnMut(WindowInsets)>>,
    /// The insets last reported through [`Self::notify_insets_changed`].
    last_insets: RefCell<WindowInsets>,
    /// Counts frames, to run the geometry check only every few of them.
    geometry_tick: Cell<u32>,
    keyboard_dismiss_callback: CallbackSlot<Box<dyn FnMut()>>,
    keyboard_dismiss_touch: Cell<Option<KeyboardDismissTouch>>,
    keyboard_height: Cell<f32>,
    /// Current mouse position (from touch)
    mouse_position: Cell<Point<Pixels>>,
    /// Current modifiers
    modifiers: Cell<Modifiers>,
    renderer: Mutex<MetalRenderer>,
    background: Cell<WindowBackgroundAppearance>,
}

impl IosWindow {
    #[allow(deprecated)] // Window construction can precede scene connection.
    pub fn new(handle: AnyWindowHandle, _params: WindowParams) -> anyhow::Result<Self> {
        // Create the window on the main screen
        let screen = IosDisplay::main();
        let screen_bounds = screen.bounds();
        let scale_factor = screen.scale();

        unsafe {
            let main_thread = MainThreadMarker::new().expect("UIKit requires the main thread");
            // Create UIWindow
            let window_scene = super::application::window_scene();
            let window_scene = window_scene.as_deref();
            let screen_obj = if let Some(scene) = window_scene {
                scene.screen()
            } else {
                UIScreen::mainScreen(main_thread)
            };
            let screen_bounds_cg = screen_obj.bounds();
            let window = if let Some(scene) = window_scene {
                let window = UIWindow::initWithWindowScene(UIWindow::alloc(main_thread), scene);
                window.setFrame(screen_bounds_cg);
                window
            } else {
                UIWindow::initWithFrame(UIWindow::alloc(main_thread), screen_bounds_cg)
            };

            let view_controller = ViewController::new(main_thread);
            let metal_frame = CGRect::new(CGPoint::ZERO, screen_bounds_cg.size);
            let view = MetalView::new(metal_frame, main_thread);

            let layer = view.layer();
            let scale = screen_obj.scale();
            layer.setContentsScale(scale);

            // Auto-resize the Metal view when the parent view changes size
            // (e.g. rotation). UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight
            view.setAutoresizingMask(
                UIViewAutoresizing::FlexibleWidth | UIViewAutoresizing::FlexibleHeight,
            );

            // Enable user interaction on the Metal view for touch handling
            view.setUserInteractionEnabled(true);
            view.setMultipleTouchEnabled(true);
            view.install_gesture_recognizers();
            super::pointer::install(&view, main_thread);

            view_controller.setView(Some(&view));

            // Set the root view controller
            window.setRootViewController(Some(&view_controller));

            // Make the window visible
            window.makeKeyAndVisible();

            // Create a hidden text input view for keyboard handling.
            // Uses our custom GPUITextInputView which implements UIKeyInput
            // so iOS actually routes keyboard text to us.
            let text_input_frame = CGRect::new(CGPoint::ZERO, CGSize::new(1.0, 1.0));
            let text_input_view = TextInputView::new(text_input_frame, main_thread);
            text_input_view.setAlpha(0.01);
            text_input_view.setUserInteractionEnabled(true);
            view.addSubview(&text_input_view);

            let edit_menu_interaction = if AnyClass::get(c"UIEditMenuInteraction").is_some() {
                let interaction = UIEditMenuInteraction::initWithDelegate(
                    UIEditMenuInteraction::alloc(main_thread),
                    None,
                );
                view.addInteraction(objc2::runtime::ProtocolObject::from_ref(&*interaction));
                Some(interaction)
            } else {
                None
            };

            let pixel_w = (screen_bounds_cg.size.width * scale) as i32;
            let pixel_h = (screen_bounds_cg.size.height * scale) as i32;
            let mut renderer = MetalRenderer::from_layer(
                MetalContext::default(),
                Retained::as_ptr(&layer)
                    .cast_mut()
                    .cast::<metal::CAMetalLayer>(),
                false,
            );
            renderer.update_drawable_size(size(DevicePixels(pixel_w), DevicePixels(pixel_h)));

            let state = IosWindowState {
                handle,
                active: Cell::new(false),
                window,
                view_controller,
                view,
                text_input_view,
                edit_menu_interaction,
                edit_menu_actions: Cell::new(EditMenuActions::default()),
                bounds: Cell::new(screen_bounds),
                scale_factor: Cell::new(scale_factor),
                input_handler: CallbackSlot::default(),
                request_frame_callback: CallbackSlot::default(),
                force_next_frame: Cell::new(true),
                input_callback: CallbackSlot::default(),
                active_status_callback: CallbackSlot::default(),
                visibility: Cell::new(WindowVisibility::Visible),
                visibility_callback: CallbackSlot::default(),
                hover_status_callback: CallbackSlot::default(),
                resize_callback: CallbackSlot::default(),
                moved_callback: CallbackSlot::default(),
                should_close_callback: CallbackSlot::default(),
                hit_test_callback: CallbackSlot::default(),
                close_callback: CallbackSlot::default(),
                appearance_changed_callback: CallbackSlot::default(),
                insets_changed_callback: CallbackSlot::default(),
                last_insets: RefCell::new(WindowInsets::default()),
                geometry_tick: Cell::new(0),
                keyboard_dismiss_callback: CallbackSlot::default(),
                keyboard_dismiss_touch: Cell::new(None),
                keyboard_height: Cell::new(0.),
                mouse_position: Cell::new(Point::default()),
                modifiers: Cell::new(Modifiers::default()),
                renderer: Mutex::new(renderer),
                background: Cell::new(WindowBackgroundAppearance::Opaque),
            };

            Ok(Self {
                state: Rc::new(state),
            })
        }
    }

    pub(crate) fn register(&self) {
        super::application::register_window(&self.state);

        let window = Rc::downgrade(&self.state);
        *self.view_controller.ivars().0.borrow_mut() = window.clone();
        *self.view.ivars().0.borrow_mut() = window.clone();
        self.text_input_view.set_window(window);

        IosWindowState::register_keyboard_observers();
    }
}

impl IosWindowState {
    pub(super) fn attach_to_scene(&self, scene: &UIWindowScene) {
        self.window.setWindowScene(Some(scene));
        self.window.makeKeyAndVisible();
        self.handle_layout_change();
    }

    fn register_keyboard_observers() {
        KEYBOARD_OBSERVERS_REGISTERED.call_once(|| unsafe {
            let notification_center = NSNotificationCenter::defaultCenter();

            let frame_change_block =
                block2::RcBlock::new(move |notification: NonNull<NSNotification>| {
                    let Some(user_info) = notification.as_ref().userInfo() else {
                        return;
                    };
                    let Some(frame_value) = user_info.objectForKey(UIKeyboardFrameEndUserInfoKey)
                    else {
                        return;
                    };
                    let Some(frame_value) = frame_value.downcast_ref::<NSValue>() else {
                        return;
                    };
                    let frame = frame_value.CGRectValue();
                    super::application::with_windows(|window| {
                        window.set_keyboard_height(frame.size.height as f32);
                    });
                });

            let hide_block = block2::RcBlock::new(move |_notification: NonNull<NSNotification>| {
                super::application::with_windows(|window| window.set_keyboard_height(0.));
            });

            notification_center.addObserverForName_object_queue_usingBlock(
                Some(UIKeyboardWillChangeFrameNotification),
                None,
                None,
                &frame_change_block,
            );
            notification_center.addObserverForName_object_queue_usingBlock(
                Some(UIKeyboardWillHideNotification),
                None,
                None,
                &hide_block,
            );
        });
    }

    pub fn dispatch_pinch(&self, position: Point<Pixels>, delta: f32, phase: TouchPhase) {
        self.mouse_position.set(position);
        self.input_callback.with(|callback| {
            callback(PlatformInput::Pinch(PinchEvent {
                position,
                delta,
                modifiers: self.modifiers.get(),
                phase,
            }))
        });
    }

    pub fn dispatch_hover(&self, position: Point<Pixels>, state: UIGestureRecognizerState) {
        self.mouse_position.set(position);
        if state == UIGestureRecognizerState::Ended || state == UIGestureRecognizerState::Cancelled {
            self.input_callback.with(|callback| {
                callback(PlatformInput::MouseExited(MouseExitEvent {
                    position,
                    pressed_button: None,
                    modifiers: self.modifiers.get(),
                }))
            });
            return;
        }
        self.input_callback.with(|callback| {
            callback(PlatformInput::MouseMove(MouseMoveEvent {
                position,
                pressed_button: None,
                modifiers: self.modifiers.get(),
            }))
        });
    }

    pub fn dispatch_pointer_scroll(
        &self,
        position: Point<Pixels>,
        delta: Point<Pixels>,
        phase: TouchPhase,
    ) {
        self.input_callback.with(|callback| {
            callback(PlatformInput::ScrollWheel(ScrollWheelEvent {
                position,
                delta: ScrollDelta::Pixels(delta),
                modifiers: self.modifiers.get(),
                touch_phase: phase,
            }))
        });
    }

    /// Delivers a UIKit touch through GPUI's platform-neutral touch API.
    pub fn handle_touch(&self, touch: &UITouch, _event: Option<&UIEvent>) {
        // GPUI timestamps delivery itself. Replaying UIKit's historical coalesced
        // samples here would make them appear simultaneous and distort fling velocity.
        let id = touch_id(touch);
        self.handle_touch_sample(touch, id);
    }

    fn handle_touch_sample(&self, touch: &UITouch, id: TouchId) {
        let position = touch_location_in_view(touch, &self.view);
        self.mouse_position.set(position);

        let event = TouchEvent {
            id,
            phase: touch_phase(touch),
            position,
            predicted_position: None,
            force: touch_force(touch),
        };
        self.handle_keyboard_dismiss_touch(&event);
        self.input_callback
            .with(|callback| callback(PlatformInput::Touch(event)));
    }

    fn handle_keyboard_dismiss_touch(&self, event: &TouchEvent) {
        if self.keyboard_height.get() <= 0. {
            self.keyboard_dismiss_touch.set(None);
            return;
        }

        match event.phase {
            TouchPhase::Started if self.keyboard_dismiss_touch.get().is_none() => {
                self.keyboard_dismiss_touch.set(Some(KeyboardDismissTouch {
                    id: event.id,
                    start_position: event.position,
                }));
            }
            TouchPhase::Moved => {
                let Some(touch) = self
                    .keyboard_dismiss_touch
                    .get()
                    .filter(|touch| touch.id == event.id)
                else {
                    return;
                };
                let delta = event.position - touch.start_position;
                if delta.y >= KEYBOARD_DISMISS_DISTANCE && delta.y.abs() > delta.x.abs() * 1.25 {
                    self.keyboard_dismiss_touch.set(None);
                    self.dismiss_keyboard();
                }
            }
            TouchPhase::Ended | TouchPhase::Cancelled => {
                if self
                    .keyboard_dismiss_touch
                    .get()
                    .is_some_and(|touch| touch.id == event.id)
                {
                    self.keyboard_dismiss_touch.set(None);
                }
            }
            TouchPhase::Started => {}
        }
    }

    pub(super) fn request_frame(&self) {
        self.sync_geometry();
        self.request_frame_callback.with(|callback| {
            let force_render = self.force_next_frame.replace(false);
            callback(RequestFrameOptions {
                force_render,
                ..Default::default()
            });
        });
    }

    /// Reconcile what GPUI was told with what UIKit currently says. UIKit does not always announce a
    /// change: a window created before its scene reports the scene's real size (a Stage Manager
    /// window that was resized before the app was launched) keeps the screen-sized frame it was
    /// given, and the window-controls insets can appear after the last layout pass. Every few
    /// frames, fit the window to its scene, then report a size or inset change GPUI missed.
    fn sync_geometry(&self) {
        let tick = self.geometry_tick.get().wrapping_add(1);
        self.geometry_tick.set(tick);
        if tick % 6 != 0 {
            return;
        }
        if let Some(scene) = super::application::window_scene() {
            #[allow(deprecated)]
            let scene_bounds = scene.coordinateSpace().bounds();
            let frame = self.window.frame();
            if (frame.size.width - scene_bounds.size.width).abs() > 0.5
                || (frame.size.height - scene_bounds.size.height).abs() > 0.5
            {
                log::debug!(
                    "GPUI iOS: fitting the window ({:?}) to its scene ({:?})",
                    frame.size,
                    scene_bounds.size
                );
                self.window.setFrame(scene_bounds);
                self.view.setNeedsLayout();
            }
        }
        let view_size = self.view.bounds().size;
        let stored = self.bounds.get().size;
        if (view_size.width as f32 - f32::from(stored.width)).abs() > 0.5
            || (view_size.height as f32 - f32::from(stored.height)).abs() > 0.5
        {
            self.handle_layout_change();
        } else if *self.last_insets.borrow() != self.current_insets() {
            self.notify_insets_changed();
        }
    }

    /// Query the safe area insets from the UIView.
    ///
    /// Returns `(top, bottom, left, right)` in logical points.
    /// These represent the areas occupied by system UI (status bar,
    /// home indicator, camera notch) that content should avoid.
    fn safe_area_insets(&self) -> (f32, f32, f32, f32) {
        let insets = self.view.safeAreaInsets();
        (
            insets.top as f32,
            insets.bottom as f32,
            insets.left as f32,
            insets.right as f32,
        )
    }

    /// Whether the window fills the whole screen. It does on an iPhone and for an iPad app that is
    /// not windowed; a Stage Manager window that has been resized (or moved) does not.
    fn covers_screen(&self) -> bool {
        let screen = IosDisplay::main().bounds().size;
        let size = self.bounds.get().size;
        size.width >= screen.width - px(1.) && size.height >= screen.height - px(1.)
    }

    /// The space the system's window controls take up over the content (`(top, bottom, left,
    /// right)`): the corner-adapted safe-area layout regions (iOS 26) minus the plain safe area.
    /// The horizontal-axis region says how far content has to move sideways to clear the
    /// controls, the vertical-axis one how far down. Zero on older systems, which lack the API.
    fn window_controls_insets(&self) -> (f32, f32, f32, f32) {
        let selector = sel!(edgeInsetsForLayoutRegion:);
        if !self.view.respondsToSelector(selector) {
            return (0., 0., 0., 0.);
        }
        let horizontal_region = UIViewLayoutRegion::safeAreaLayoutRegionWithCornerAdaptation(
            UIViewLayoutRegionAdaptivityAxis::Horizontal,
        );
        let vertical_region = UIViewLayoutRegion::safeAreaLayoutRegionWithCornerAdaptation(
            UIViewLayoutRegionAdaptivityAxis::Vertical,
        );
        // SAFETY: the selector was just checked; it takes a layout region and returns UIEdgeInsets.
        let horizontal: UIEdgeInsets =
            unsafe { msg_send![&*self.view, edgeInsetsForLayoutRegion: &*horizontal_region] };
        let vertical: UIEdgeInsets =
            unsafe { msg_send![&*self.view, edgeInsetsForLayoutRegion: &*vertical_region] };
        let safe = self.view.safeAreaInsets();
        (
            (vertical.top - safe.top).max(0.) as f32,
            (vertical.bottom - safe.bottom).max(0.) as f32,
            (horizontal.left - safe.left).max(0.) as f32,
            (horizontal.right - safe.right).max(0.) as f32,
        )
    }

    fn current_insets(&self) -> WindowInsets {
        let (top, bottom, left, right) = self.safe_area_insets();
        let (controls_top, controls_bottom, controls_left, controls_right) =
            self.window_controls_insets();
        WindowInsets {
            window_controls: Edges {
                top: px(controls_top),
                right: px(controls_right),
                bottom: px(controls_bottom),
                left: px(controls_left),
            },
            safe_area: Edges {
                top: px(top),
                right: px(right),
                bottom: px(bottom),
                left: px(left),
            },
            ime: Edges {
                bottom: px(self.keyboard_height.get()),
                ..Default::default()
            },
        }
    }

    fn notify_insets_changed(&self) {
        let insets = self.current_insets();
        *self.last_insets.borrow_mut() = insets.clone();
        // Fires on every layout pass (each frame while a window is being resized), so trace only.
        log::trace!(
            "GPUI iOS: insets — safe {:?}, window controls {:?}; window {:?} on screen {:?} (covers: {})",
            insets.safe_area,
            insets.window_controls,
            self.bounds.get().size,
            IosDisplay::main().bounds().size,
            self.covers_screen()
        );
        self.insets_changed_callback.with(|callback| callback(insets));
    }

    fn notify_appearance_changed(&self) {
        self.appearance_changed_callback.with(|callback| callback());
    }

    fn set_keyboard_height(&self, height: f32) {
        let height = height.max(0.);
        if (self.keyboard_height.get() - height).abs() <= 0.5 {
            return;
        }
        self.keyboard_height.set(height);
        if height <= 0. {
            self.keyboard_dismiss_touch.set(None);
        }
        self.notify_insets_changed();
    }

    pub fn show_keyboard(&self) {
        self.text_input_view.set_keyboard_visible(true);
    }

    pub fn hide_keyboard(&self) {
        self.text_input_view.set_keyboard_visible(false);
    }

    fn dismiss_keyboard(&self) {
        self.hide_keyboard();
        self.keyboard_dismiss_callback.with(|callback| callback());
    }

    pub(super) fn with_input_handler<R>(
        &self,
        callback: impl FnOnce(&mut PlatformInputHandler) -> R,
    ) -> Option<R> {
        self.input_handler.with(callback)
    }

    fn dispatch_edit_menu_shortcut(&self, key: &str) {
        let event = PlatformInput::KeyDown(gpui::KeyDownEvent {
            keystroke: gpui::Keystroke {
                modifiers: Modifiers {
                    platform: true,
                    ..Modifiers::default()
                },
                key: key.to_string(),
                key_char: Some(key.to_string()),
            },
            is_held: false,
            prefer_character_input: false,
        });
        self.input_callback.with(|callback| callback(event));
    }

    /// Notify the window when its UIKit scene becomes active or inactive.
    /// Whether the main menu's Edit commands apply: a focused text input that takes text, with a
    /// selection for cut and copy. (The touch edit menu supplies its own availability instead.)
    fn can_edit_from_menu_bar(&self, action: Sel) -> bool {
        let needs_selection = action == sel!(cut:) || action == sel!(copy:);
        if !needs_selection && action != sel!(paste:) && action != sel!(selectAll:) {
            return false;
        }
        self.input_handler
            .with(|handler| {
                handler.query_accepts_focused_text_input()
                    && (!needs_selection
                        || handler
                            .selected_text_range(false)
                            .is_some_and(|selection| !selection.range.is_empty()))
            })
            .unwrap_or(false)
    }

    pub(super) fn handle(&self) -> AnyWindowHandle {
        self.handle
    }

    pub(super) fn is_active(&self) -> bool {
        self.active.get()
    }

    pub fn notify_active_status_change(&self, is_active: bool) {
        log::info!("GPUI iOS: Window active status changed to: {}", is_active);
        self.active.set(is_active);

        self.active_status_callback
            .with(|callback| callback(is_active));
    }

    pub(crate) fn notify_visibility_change(&self, visibility: WindowVisibility) {
        if self.visibility.replace(visibility) != visibility {
            self.visibility_callback
                .with(|callback| callback(visibility));
        }
    }

    pub fn handle_layout_change(&self) {
        let view_bounds = self.view.bounds();
        let scale = self.window.screen().scale();

        let new_w = view_bounds.size.width as f32;
        let new_h = view_bounds.size.height as f32;
        let new_scale = scale as f32;

        let old_bounds = self.bounds.get();
        let old_scale = self.scale_factor.get();

        let new_size = size(px(new_w), px(new_h));
        self.notify_insets_changed();

        if old_bounds.size == new_size && (old_scale - new_scale).abs() < 0.01 {
            return;
        }

        log::trace!(
            "GPUI iOS: Layout changed — {:?} @{:.1}x → {:?} @{:.1}x",
            old_bounds.size,
            old_scale,
            new_size,
            new_scale,
        );

        // Update stored bounds (in logical pixels, matching GPUI convention).
        let new_bounds = Bounds {
            origin: Default::default(),
            size: new_size,
        };
        self.bounds.set(new_bounds);
        self.scale_factor.set(new_scale);
        // Update the Metal layer's contentsScale so the drawable has the
        // correct pixel dimensions.
        self.view.layer().setContentsScale(scale);

        let pixel_w = (new_w * new_scale) as i32;
        let pixel_h = (new_h * new_scale) as i32;
        self.renderer
            .lock()
            .update_drawable_size(size(DevicePixels(pixel_w), DevicePixels(pixel_h)));

        self.resize_callback
            .with(|callback| callback(new_size, new_scale));
    }
}

impl Drop for IosWindow {
    fn drop(&mut self) {
        super::application::unregister_window(&self.state);

        *self.view_controller.ivars().0.borrow_mut() = Weak::new();
        *self.view.ivars().0.borrow_mut() = Weak::new();
        self.text_input_view.set_window(Weak::new());
        self.text_input_view.removeFromSuperview();
        if let Some(interaction) = &self.edit_menu_interaction {
            self.view
                .removeInteraction(objc2::runtime::ProtocolObject::from_ref(&**interaction));
        }
    }
}

impl HasWindowHandle for IosWindow {
    fn window_handle(
        &self,
    ) -> std::result::Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError>
    {
        let view = NonNull::new(Retained::as_ptr(&self.view).cast_mut().cast::<c_void>())
            .ok_or(raw_window_handle::HandleError::Unavailable)?;
        let handle = UiKitWindowHandle::new(view);
        Ok(unsafe { raw_window_handle::WindowHandle::borrow_raw(handle.into()) })
    }
}

impl HasDisplayHandle for IosWindow {
    fn display_handle(
        &self,
    ) -> std::result::Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError>
    {
        let handle = UiKitDisplayHandle::new();
        Ok(unsafe { raw_window_handle::DisplayHandle::borrow_raw(handle.into()) })
    }
}

impl PlatformWindow for IosWindow {
    fn bounds(&self) -> Bounds<Pixels> {
        self.bounds.get()
    }

    fn is_maximized(&self) -> bool {
        self.covers_screen()
    }

    fn window_bounds(&self) -> WindowBounds {
        WindowBounds::Fullscreen(self.bounds.get())
    }

    fn content_size(&self) -> Size<Pixels> {
        self.bounds.get().size
    }

    fn resize(&mut self, _size: Size<Pixels>) {
        // iOS windows cannot be resized programmatically
    }

    fn scale_factor(&self) -> f32 {
        self.scale_factor.get()
    }

    fn appearance(&self) -> WindowAppearance {
        match unsafe { self.view.traitCollection().userInterfaceStyle() } {
            UIUserInterfaceStyle::Dark => WindowAppearance::Dark,
            _ => WindowAppearance::Light,
        }
    }

    fn display(&self) -> Option<Rc<dyn PlatformDisplay>> {
        Some(Rc::new(IosDisplay::main()))
    }

    fn mouse_position(&self) -> Point<Pixels> {
        self.mouse_position.get()
    }

    fn modifiers(&self) -> Modifiers {
        self.modifiers.get()
    }

    fn capslock(&self) -> Capslock {
        // Would need to check UIKeyModifierFlags
        Capslock { on: false }
    }

    fn set_input_handler(&mut self, input_handler: PlatformInputHandler) {
        self.input_handler.set(input_handler);
        self.text_input_view.refresh_keyboard();
    }

    fn take_input_handler(&mut self) -> Option<PlatformInputHandler> {
        let handler = self.input_handler.take();
        self.text_input_view.refresh_keyboard();
        handler
    }

    fn prompt(
        &self,
        level: PromptLevel,
        msg: &str,
        detail: Option<&str>,
        answers: &[PromptButton],
    ) -> Option<futures::channel::oneshot::Receiver<usize>> {
        let mtm = MainThreadMarker::new()?;
        let presenter = super::IosPlatform::presented_view_controller()?;
        let (sender, receiver) = futures::channel::oneshot::channel();
        let sender = Rc::new(RefCell::new(Some(sender)));

        let alert = UIAlertController::alertControllerWithTitle_message_preferredStyle(
            Some(&NSString::from_str(msg)),
            detail.map(NSString::from_str).as_deref(),
            UIAlertControllerStyle::Alert,
            mtm,
        );
        // Only the first button of a critical prompt is the destructive one (the app lists the
        // confirming action first); an explicit cancel button gets the system's cancel styling.
        for (index, answer) in answers.iter().enumerate() {
            let style = if answer.is_cancel() {
                UIAlertActionStyle::Cancel
            } else if level == PromptLevel::Critical && index == 0 {
                UIAlertActionStyle::Destructive
            } else {
                UIAlertActionStyle::Default
            };
            let sender = sender.clone();
            let handler = block2::RcBlock::new(move |_action: NonNull<UIAlertAction>| {
                if let Some(sender) = sender.borrow_mut().take() {
                    if sender.send(index).is_err() {
                        log::debug!("GPUI iOS: prompt receiver was dropped");
                    }
                }
            });
            let action = UIAlertAction::actionWithTitle_style_handler(
                Some(&NSString::from_str(answer.label())),
                style,
                Some(&handler),
                mtm,
            );
            alert.addAction(&action);
        }
        presenter.presentViewController_animated_completion(&alert, true, None);
        Some(receiver)
    }

    fn activate(&self) {
        self.window.makeKeyAndVisible();
    }

    fn is_active(&self) -> bool {
        self.window.isKeyWindow()
    }

    fn is_hovered(&self) -> bool {
        // Hover isn't really applicable on iOS
        false
    }

    fn set_title(&mut self, _title: &str) {
        // iOS apps don't have window titles
    }

    fn background_appearance(&self) -> WindowBackgroundAppearance {
        self.background.get()
    }

    /// Anything but `Opaque` makes the Metal layer transparent so views placed beneath it (an
    /// app's own `UIVisualEffectView`s) show through where GPUI paints nothing. iOS has no window
    /// blur of its own, so `Blurred` and the Windows-only materials are treated as `Transparent`.
    fn set_background_appearance(&self, background_appearance: WindowBackgroundAppearance) {
        self.background.set(background_appearance);
        let transparent = background_appearance != WindowBackgroundAppearance::Opaque;
        self.renderer.lock().update_transparency(transparent);
        self.view.setOpaque(!transparent);
        self.force_next_frame.set(true);
    }

    fn minimize(&self) {
        // iOS apps cannot be minimized
    }

    fn zoom(&self) {
        // iOS apps cannot be zoomed
    }

    fn toggle_fullscreen(&self) {
        // iOS apps are always fullscreen
    }

    fn is_fullscreen(&self) -> bool {
        self.covers_screen()
    }

    fn on_request_frame(&self, callback: Box<dyn FnMut(RequestFrameOptions)>) {
        self.request_frame_callback.set(callback);
    }

    fn on_input(&self, callback: Box<dyn FnMut(PlatformInput) -> DispatchEventResult>) {
        self.input_callback.set(callback);
    }

    fn on_active_status_change(&self, callback: Box<dyn FnMut(bool)>) {
        self.active_status_callback.set(callback);
    }

    fn visibility(&self) -> WindowVisibility {
        self.visibility.get()
    }

    fn on_visibility_change(&self, callback: Box<dyn FnMut(WindowVisibility)>) {
        self.visibility_callback.set(callback);
    }

    fn on_hover_status_change(&self, callback: Box<dyn FnMut(bool)>) {
        self.hover_status_callback.set(callback);
    }

    fn on_resize(&self, callback: Box<dyn FnMut(Size<Pixels>, f32)>) {
        self.resize_callback.set(callback);
    }

    fn on_moved(&self, callback: Box<dyn FnMut()>) {
        self.moved_callback.set(callback);
    }

    fn on_should_close(&self, callback: Box<dyn FnMut() -> bool>) {
        self.should_close_callback.set(callback);
    }

    fn on_hit_test_window_control(&self, callback: Box<dyn FnMut() -> Option<WindowControlArea>>) {
        self.hit_test_callback.set(callback);
    }

    fn on_close(&self, callback: Box<dyn FnOnce()>) {
        self.close_callback.set(callback);
    }

    fn on_appearance_changed(&self, callback: Box<dyn FnMut()>) {
        self.appearance_changed_callback.set(callback);
    }

    fn draw(&self, scene: &Scene) {
        self.renderer.lock().draw(scene);
    }

    fn sprite_atlas(&self) -> Arc<dyn PlatformAtlas> {
        self.renderer.lock().sprite_atlas().clone()
    }

    fn is_subpixel_rendering_supported(&self) -> bool {
        false
    }

    fn gpu_specs(&self) -> Option<GpuSpecs> {
        None
    }

    fn update_ime_position(&self, _bounds: Bounds<Pixels>) {
        // iOS handles IME positioning automatically
    }

    fn insets(&self) -> WindowInsets {
        self.current_insets()
    }

    fn on_insets_changed(&self, callback: Box<dyn FnMut(WindowInsets)>) {
        self.insets_changed_callback.set(callback);
    }

    fn show_soft_keyboard(&self) {
        self.show_keyboard();
    }

    fn hide_soft_keyboard(&self) {
        self.hide_keyboard();
    }

    fn set_keyboard_dismiss_handler(&self, callback: Box<dyn FnMut()>) {
        self.keyboard_dismiss_callback.set(callback);
    }

    fn show_edit_menu(&self, position: Point<Pixels>, actions: EditMenuActions) -> bool {
        let Some(interaction) = &self.edit_menu_interaction else {
            return false;
        };

        self.edit_menu_actions.set(actions);
        unsafe {
            let source_point = CGPoint {
                x: f64::from(position.x),
                y: f64::from(position.y),
            };
            let configuration = UIEditMenuConfiguration::configurationWithIdentifier_sourcePoint(
                None,
                source_point,
                self.view.mtm(),
            );
            interaction.dismissMenu();
            interaction.presentEditMenuWithConfiguration(&configuration);
        }
        true
    }

    fn text_input_state_changed(&self, change: TextInputStateChange) {
        self.text_input_view.state_changed(change);
    }

    fn set_text_input_configuration(&mut self, configuration: TextInputConfiguration) {
        self.text_input_view.set_configuration(&configuration);
    }
}

#[cfg(test)]
mod tests {
    use super::CallbackSlot;

    #[test]
    fn callback_is_unavailable_during_dispatch_and_restored_afterward() {
        let slot = CallbackSlot::default();
        slot.set(1);
        assert_eq!(
            slot.with(|value| {
                *value += 1;
                assert!(
                    slot.with(|_| panic!("reentered the same handler"))
                        .is_none()
                );
                *value
            }),
            Some(2)
        );
        assert_eq!(slot.with(|value| *value), Some(2));
    }

    #[test]
    fn callback_replacement_during_dispatch_is_preserved() {
        let slot = CallbackSlot::default();
        slot.set(1);
        slot.with(|_| {
            slot.set(2);
            slot.with(|value| *value += 1);
        });
        assert_eq!(slot.take(), Some(3));
    }

    #[test]
    fn withdrawing_an_in_flight_handler_does_not_restore_it() {
        let slot = CallbackSlot::default();
        slot.set(1);
        slot.with(|_| assert!(slot.take().is_none()));
        assert!(
            slot.with(|_| panic!("restored a withdrawn handler"))
                .is_none()
        );
    }

    #[test]
    fn removing_a_replacement_does_not_resurrect_the_previous_handler() {
        let slot = CallbackSlot::default();
        slot.set(1);
        slot.with(|_| {
            slot.set(2);
            assert_eq!(slot.take(), Some(2));
        });
        assert!(slot.take().is_none());
    }
}

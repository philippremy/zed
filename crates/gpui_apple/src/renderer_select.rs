//! Seam for choosing a GPU backend at renderer-construction time.
//!
//! `gpui_macos` never names `MetalRenderer` directly — every call goes
//! through this module via the `renderer` alias set up in `gpui_macos.rs`.
//! That indirection is what lets a second backend be added here later
//! without touching window/display-link code at all: this module, and the
//! `Renderer` enum in particular, is the only seam that needs to change.
//!
//! Metal 4 support is not implemented yet. `metal4_capability::metal4_available()`
//! is real and already exercised by tests, but `new_renderer` unconditionally
//! constructs the Metal 3 backend for now — there is no `V4` variant to
//! select. Wiring one in is tracked separately and should only require a new
//! match arm here plus a new renderer module, per the doc comment above.

use crate::metal_atlas::MetalAtlas;
use crate::metal_renderer::MetalRenderer;
use gpui::{DevicePixels, Scene, Size};
use std::ffi::c_void;
use std::sync::Arc;

#[cfg(any(test, feature = "test-support"))]
use anyhow::Result;
#[cfg(any(test, feature = "test-support"))]
use image::RgbaImage;

pub type Context = crate::metal_renderer::Context;

pub enum Renderer {
    V3(MetalRenderer),
}

pub unsafe fn new_renderer(
    context: Context,
    native_window: *mut c_void,
    native_view: *mut c_void,
    bounds: Size<f32>,
    transparent: bool,
) -> Renderer {
    // Metal 4 hardware/OS support is already detected here so the signal is
    // visible in logs ahead of a `V4` variant existing to act on it — there
    // is no dispatch to do yet, this always constructs the Metal 3 backend.
    log::debug!(
        "gpu backend: metal4 available = {}, selecting metal3 (metal4 renderer not implemented yet)",
        crate::metal4_capability::metal4_available()
    );

    Renderer::V3(unsafe {
        crate::metal_renderer::new_renderer(context, native_window, native_view, bounds, transparent)
    })
}

impl Renderer {
    pub fn draw(&mut self, scene: &Scene) {
        match self {
            Self::V3(renderer) => renderer.draw(scene),
        }
    }

    pub fn destroy(&self) {
        match self {
            Self::V3(renderer) => renderer.destroy(),
        }
    }

    pub fn sprite_atlas(&self) -> &Arc<MetalAtlas> {
        match self {
            Self::V3(renderer) => renderer.sprite_atlas(),
        }
    }

    pub fn update_transparency(&mut self, transparent: bool) {
        match self {
            Self::V3(renderer) => renderer.update_transparency(transparent),
        }
    }

    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        match self {
            Self::V3(renderer) => renderer.update_drawable_size(size),
        }
    }

    pub fn set_presents_with_transaction(&mut self, presents_with_transaction: bool) {
        match self {
            Self::V3(renderer) => renderer.set_presents_with_transaction(presents_with_transaction),
        }
    }

    pub fn layer(&self) -> Option<&metal::MetalLayerRef> {
        match self {
            Self::V3(renderer) => renderer.layer(),
        }
    }

    pub fn layer_ptr(&self) -> *mut metal::CAMetalLayer {
        match self {
            Self::V3(renderer) => renderer.layer_ptr(),
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn render_to_image(&mut self, scene: &Scene) -> Result<RgbaImage> {
        match self {
            Self::V3(renderer) => renderer.render_to_image(scene),
        }
    }
}

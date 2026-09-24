//! Seam for choosing a GPU backend at renderer-construction time.
//!
//! `gpui_macos` never names `MetalRenderer` directly — every call goes
//! through this module via the `renderer` alias set up in `gpui_macos.rs`.
//! That indirection is what lets a second backend be added here later
//! without touching window/display-link code at all: this module, and the
//! `Renderer` enum in particular, is the only seam that needs to change.
//!
//! `V4(Metal4Renderer)` exists but is quads-only (see `metal4_renderer`'s
//! module doc for scope and the reasoning behind it) and is never selected
//! by default: it only activates when `metal4_available()` is true **and**
//! `DTB_KE_GPU_BACKEND=metal4` is set, an opt-in for testing. Every other
//! case constructs Metal 3, unchanged from before `V4` existed.

use crate::metal4_renderer::Metal4Renderer;
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
    V4(Metal4Renderer),
}

pub unsafe fn new_renderer(
    context: Context,
    native_window: *mut c_void,
    native_view: *mut c_void,
    bounds: Size<f32>,
    transparent: bool,
) -> Renderer {
    let use_metal4 = crate::metal4_capability::metal4_available();
    log::debug!(
        "gpu backend: using {}",
        if use_metal4 { "Metal 4" } else { "Metal 3" }
    );

    if use_metal4 {
        Renderer::V4(Metal4Renderer::new(transparent))
    } else {
        Renderer::V3(unsafe {
            crate::metal_renderer::new_renderer(
                context,
                native_window,
                native_view,
                bounds,
                transparent,
            )
        })
    }
}

impl Renderer {
    pub fn draw(&mut self, scene: &Scene) {
        match self {
            Self::V3(renderer) => renderer.draw(scene),
            Self::V4(renderer) => renderer.draw(scene),
        }
    }

    pub fn destroy(&self) {
        match self {
            Self::V3(renderer) => renderer.destroy(),
            Self::V4(renderer) => renderer.destroy(),
        }
    }

    pub fn sprite_atlas(&self) -> &Arc<MetalAtlas> {
        match self {
            Self::V3(renderer) => renderer.sprite_atlas(),
            Self::V4(renderer) => renderer.sprite_atlas(),
        }
    }

    pub fn update_transparency(&mut self, transparent: bool) {
        match self {
            Self::V3(renderer) => renderer.update_transparency(transparent),
            Self::V4(renderer) => renderer.update_transparency(transparent),
        }
    }

    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        match self {
            Self::V3(renderer) => renderer.update_drawable_size(size),
            Self::V4(renderer) => renderer.update_drawable_size(size),
        }
    }

    pub fn set_presents_with_transaction(&mut self, presents_with_transaction: bool) {
        match self {
            Self::V3(renderer) => renderer.set_presents_with_transaction(presents_with_transaction),
            Self::V4(renderer) => renderer.set_presents_with_transaction(presents_with_transaction),
        }
    }

    pub fn layer(&self) -> Option<&metal::MetalLayerRef> {
        match self {
            Self::V3(renderer) => renderer.layer(),
            Self::V4(renderer) => renderer.layer(),
        }
    }

    pub fn layer_ptr(&self) -> *mut metal::CAMetalLayer {
        match self {
            Self::V3(renderer) => renderer.layer_ptr(),
            Self::V4(renderer) => renderer.layer_ptr(),
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn render_to_image(&mut self, scene: &Scene) -> Result<RgbaImage> {
        match self {
            Self::V3(renderer) => renderer.render_to_image(scene),
            Self::V4(_renderer) => {
                anyhow::bail!("render_to_image is not implemented for the Metal 4 renderer yet")
            }
        }
    }
}

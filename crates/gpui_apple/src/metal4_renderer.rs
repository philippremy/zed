//! A first-milestone Metal 4 renderer: solid-color quads only.
//!
//! Ported from `metal_renderer.rs`'s device/layer setup (unchanged — Metal 4
//! doesn't touch `CAMetalLayer`/drawable acquisition at all) but the
//! command-submission and resource-binding halves are new: Metal 4 drops
//! automatic resource tracking entirely ("In Metal 4, the framework
//! considers all resources untracked" — Apple's "Understanding the Metal 4
//! core API"), so every buffer/texture the GPU touches must be explicitly
//! placed in an `MTLResidencySet`, and bindings go through an
//! `MTL4ArgumentTable` (a GPU address / `MTLResourceID` table) rather than
//! per-call `setVertexBuffer:`/`setFragmentTexture:`. The quad vertex/
//! fragment shaders themselves are untouched and reused byte-for-byte from
//! the existing `shaders.metallib` — `[[buffer(N)]]` argument-table slots
//! and classic bind-point indices are the same binding namespace from the
//! shader's perspective, only how the CPU side populates slot N differs.
//!
//! Deliberately simplified vs. a production implementation: one command
//! allocator and one command buffer, fully serialized (each frame waits for
//! the GPU to finish before the next begins, via an `MTLSharedEvent`) rather
//! than Apple's recommended triple-buffered allocator rotation. That
//! sacrifices pipelining for far less state to get wrong in a first,
//! correctness-focused milestone — see `draw()`.
//!
//! Verified against Apple's own "Drawing a Triangle with Metal 4" sample
//! (`Metal4Renderer.m` / `+Setup.m` / `+Compilation.m` / `+Encoding.m`) for
//! the parts that aren't otherwise documented in the Metal 4 API reference:
//! command-buffer reuse via `beginCommandBufferWithAllocator:` after commit,
//! queue-level (not command-buffer-level) residency via `addResidencySet:`,
//! and the drawable present choreography (`waitForDrawable:` before commit,
//! `signalDrawable:` after commit, then `[drawable present]`).

use crate::metal_atlas::MetalAtlas;
use foreign_types::{ForeignType, ForeignTypeRef};
use gpui::{DevicePixels, Quad, Scene, Size};
use objc::{msg_send, sel, sel_impl};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4BlendState, MTL4CommandAllocator,
    MTL4CommandBuffer, MTL4CommandEncoder, MTL4CommandQueue, MTL4Compiler, MTL4CompilerDescriptor,
    MTL4LibraryFunctionDescriptor, MTL4RenderCommandEncoder, MTL4RenderPassDescriptor,
    MTL4RenderPipelineDescriptor, MTLBuffer, MTLClearColor, MTLDevice, MTLLoadAction,
    MTLPixelFormat, MTLPrimitiveType, MTLRenderPipelineState, MTLRenderStages, MTLResidencySet,
    MTLResidencySetDescriptor, MTLResourceOptions, MTLSharedEvent, MTLStoreAction,
};
use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use std::mem;
use std::ptr::NonNull;

/// Bridges a raw pointer from the legacy `metal`/`objc` crates (both of
/// which ultimately wrap the same `id`/`*mut objc::runtime::Object`) into an
/// owned objc2 handle, retaining it. `metal`'s `ForeignType::as_ptr()` and
/// objc2's `AnyObject` are both thin wrappers over the same Objective-C
/// object representation, so this is a same-object reinterpretation, not a
/// conversion — the two binding crates just don't know about each other.
unsafe fn bridge_retain<T: objc2::Message>(ptr: *mut c_void) -> Retained<T> {
    let ptr = ptr as *mut T;
    unsafe { Retained::retain(ptr) }.expect("bridged Objective-C pointer was nil")
}

/// Same bridge, but for methods that hand back an autoreleased (not
/// pre-retained) reference — e.g. a `msg_send!` call with no `new`/`copy`/
/// `alloc`/`init`/`mutableCopy` prefix in its selector.
unsafe fn bridge_retain_autoreleased<T: objc2::Message>(ptr: *mut c_void) -> Retained<T> {
    let ptr = ptr as *mut T;
    unsafe { Retained::retain_autoreleased(ptr) }.expect("bridged Objective-C pointer was nil")
}

const MAX_QUADS_PER_FRAME_INITIAL: usize = 256;

#[repr(C)]
struct ViewportSize {
    width: f32,
    height: f32,
}

pub struct Metal4Renderer {
    /// Kept alive for the renderer's lifetime even though nothing reads it
    /// directly after `new_internal` — `mtl4_device` is a bridged retain of
    /// this exact object (see `bridge_retain`), and `layer.set_device`
    /// separately retains it too, but holding our own reference here keeps
    /// the ownership story explicit rather than relying on those.
    #[allow(dead_code)]
    device: metal::Device,
    layer: Option<metal::MetalLayer>,
    opaque: bool,
    /// Never populated (quads don't sample the atlas) — exists only so
    /// `Renderer::sprite_atlas()` has something to return from either variant.
    sprite_atlas: std::sync::Arc<MetalAtlas>,

    mtl4_device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    argument_table: Retained<ProtocolObject<dyn MTL4ArgumentTable>>,
    /// Long-lived residency set for everything except the drawable itself
    /// (the drawable's own residency is `CAMetalLayer.residencySet`, added
    /// to the queue separately in `new` — see the doc comment up top).
    residency_set: Retained<ProtocolObject<dyn MTLResidencySet>>,
    pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    unit_vertices: Retained<ProtocolObject<dyn MTLBuffer>>,
    viewport_size_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Reallocated (and re-added to the residency set) whenever a frame
    /// needs more quads than the current capacity.
    quads_buffer: RefCell<(Retained<ProtocolObject<dyn MTLBuffer>>, usize)>,
    /// Serializes frames: signalled after each commit, waited on before the
    /// next frame reuses the (sole) allocator/command buffer/quads buffer.
    shared_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    frame_number: Cell<u64>,
}

impl Metal4Renderer {
    pub fn new(transparent: bool) -> Self {
        let device = Self::create_device();

        let layer = metal::MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        layer.set_opaque(!transparent);
        layer.set_maximum_drawable_count(3);
        unsafe {
            let _: () = msg_send![&*layer, setAllowsNextDrawableTimeout: cocoa::base::NO];
            let _: () = msg_send![&*layer, setNeedsDisplayOnBoundsChange: cocoa::base::YES];
            let _: () = msg_send![
                &*layer,
                setAutoresizingMask: cocoa::quartzcore::AutoresizingMask::WIDTH_SIZABLE
                    | cocoa::quartzcore::AutoresizingMask::HEIGHT_SIZABLE
            ];
        }

        Self::new_internal(device, Some(layer), !transparent)
    }

    fn create_device() -> metal::Device {
        if let Some(d) = metal::Device::all()
            .into_iter()
            .min_by_key(|d| (d.is_removable(), !d.is_low_power()))
        {
            d
        } else {
            metal::Device::system_default().unwrap_or_else(|| {
                log::error!("metal4: unable to access a compatible graphics device");
                std::process::exit(1);
            })
        }
    }

    fn new_internal(
        device: metal::Device,
        layer: Option<metal::MetalLayer>,
        opaque: bool,
    ) -> Self {

        // SAFETY: `device`'s raw pointer is a live, retained `id<MTLDevice>`
        // for as long as `device` itself is alive; we take our own retain on
        // it below and hold it for the renderer's whole lifetime.
        let mtl4_device: Retained<ProtocolObject<dyn MTLDevice>> =
            unsafe { bridge_retain(device.as_ptr() as *mut c_void) };

        let is_apple_gpu = device.supports_family(metal::MTLGPUFamily::Apple1);
        let sprite_atlas = std::sync::Arc::new(MetalAtlas::new(device.clone(), is_apple_gpu));

        let queue = mtl4_device
            .newMTL4CommandQueue()
            .expect("metal4: device could not create an MTL4CommandQueue");
        let command_buffer = mtl4_device
            .newCommandBuffer()
            .expect("metal4: device could not create an MTL4CommandBuffer");
        let allocator = mtl4_device
            .newCommandAllocator()
            .expect("metal4: device could not create an MTL4CommandAllocator");

        let argument_table = {
            let descriptor = unsafe { MTL4ArgumentTableDescriptor::new() };
            unsafe { descriptor.setMaxBufferBindCount(3) };
            mtl4_device
                .newArgumentTableWithDescriptor_error(&descriptor)
                .expect("metal4: device could not create an MTL4ArgumentTable")
        };

        let residency_set = {
            let descriptor = unsafe { MTLResidencySetDescriptor::new() };
            mtl4_device
                .newResidencySetWithDescriptor_error(&descriptor)
                .expect("metal4: device could not create an MTLResidencySet")
        };

        let shared_event = mtl4_device
            .newSharedEvent()
            .expect("metal4: device could not create an MTLSharedEvent");
        unsafe { shared_event.setSignaledValue(0) };

        // The unit quad (two triangles covering [0,1]x[0,1]) — identical
        // data/layout to MetalRenderer's `unit_vertices`, just written
        // through objc2-metal instead of the legacy `metal` crate.
        let unit_vertices_data: [[f32; 2]; 6] = [
            [0., 0.],
            [1., 0.],
            [0., 1.],
            [0., 1.],
            [1., 0.],
            [1., 1.],
        ];
        let unit_vertices = Self::new_buffer_with_data(
            &mtl4_device,
            unit_vertices_data.as_ptr() as *const c_void,
            mem::size_of_val(&unit_vertices_data),
        );

        let viewport_size_buffer = Self::new_buffer(&mtl4_device, mem::size_of::<ViewportSize>());

        let quads_buffer = Self::new_buffer(
            &mtl4_device,
            mem::size_of::<Quad>() * MAX_QUADS_PER_FRAME_INITIAL,
        );

        unsafe {
            residency_set.addAllocation(unit_vertices.as_ref());
            residency_set.addAllocation(viewport_size_buffer.as_ref());
            residency_set.addAllocation(quads_buffer.as_ref());
            residency_set.commit();
            residency_set.requestResidency();
        }
        unsafe { queue.addResidencySet(&residency_set) };

        // The drawable's own texture needs to be resident too, via the
        // layer's own residency set — a real, separate `MTLResidencySet`
        // Metal 4-aware `CAMetalLayer`s expose (confirmed via Apple's
        // "drawing-a-triangle-with-metal-4" sample, which adds it to the
        // queue the exact same way). The legacy `metal` crate has no
        // binding for this new property, so it's fetched via a raw
        // `objc` message send and bridged the same way as the device above.
        if let Some(layer) = &layer {
            let layer_residency_set_ptr: *mut c_void =
                unsafe { msg_send![&**layer, residencySet] };
            if !layer_residency_set_ptr.is_null() {
                let layer_residency_set: Retained<ProtocolObject<dyn MTLResidencySet>> =
                    unsafe { bridge_retain_autoreleased(layer_residency_set_ptr) };
                unsafe { queue.addResidencySet(&layer_residency_set) };
            } else {
                log::warn!(
                    "metal4: CAMetalLayer.residencySet was nil — the drawable's own texture may not be resident"
                );
            }
        }

        let pipeline_state = Self::compile_quad_pipeline(&mtl4_device, &device);

        Self {
            device,
            layer,
            opaque,
            sprite_atlas,
            mtl4_device,
            queue,
            command_buffer,
            allocator,
            argument_table,
            residency_set,
            pipeline_state,
            unit_vertices,
            viewport_size_buffer,
            quads_buffer: RefCell::new((quads_buffer, MAX_QUADS_PER_FRAME_INITIAL)),
            shared_event,
            frame_number: Cell::new(0),
        }
    }

    fn new_buffer(
        device: &ProtocolObject<dyn MTLDevice>,
        length: usize,
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        unsafe {
            device.newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)
        }
        .expect("metal4: device could not allocate an MTLBuffer")
    }

    fn new_buffer_with_data(
        device: &ProtocolObject<dyn MTLDevice>,
        bytes: *const c_void,
        length: usize,
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let buffer = Self::new_buffer(device, length);
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes as *const u8,
                buffer.contents().as_ptr() as *mut u8,
                length,
            );
        }
        buffer
    }

    fn compile_quad_pipeline(
        mtl4_device: &ProtocolObject<dyn MTLDevice>,
        legacy_device: &metal::Device,
    ) -> Retained<ProtocolObject<dyn MTLRenderPipelineState>> {
        // Reuse the exact same, unmodified shaders.metal MetalRenderer
        // builds from — the quad_vertex/quad_fragment functions'
        // `[[buffer(N)]]` argument-table slots are the same binding
        // namespace regardless of whether the CPU side populated them via
        // classic setVertexBuffer:/setFragmentBuffer: or an MTL4ArgumentTable.
        // Mirrors metal_renderer.rs's own runtime_shaders/precompiled split
        // (this build enables the `runtime_shaders` feature, so only that
        // branch is actually exercised here, but both are kept for parity).
        #[cfg(feature = "runtime_shaders")]
        let legacy_library = {
            const SHADERS_SOURCE_FILE: &str =
                include_str!(concat!(env!("OUT_DIR"), "/stitched_shaders.metal"));
            legacy_device
                .new_library_with_source(SHADERS_SOURCE_FILE, &metal::CompileOptions::new())
                .expect("metal4: error building metal library")
        };
        #[cfg(not(feature = "runtime_shaders"))]
        let legacy_library = {
            const SHADERS_METALLIB: &[u8] =
                include_bytes!(concat!(env!("OUT_DIR"), "/shaders.metallib"));
            legacy_device
                .new_library_with_data(SHADERS_METALLIB)
                .expect("metal4: error building metal library")
        };
        // Bridge the legacy `metal::Library` into an objc2 `MTLLibrary` the
        // same way as the device — it's the same underlying `id<MTLLibrary>`.
        let library: Retained<ProtocolObject<dyn objc2_metal::MTLLibrary>> =
            unsafe { bridge_retain(legacy_library.as_ptr() as *mut c_void) };

        let compiler = {
            let descriptor = unsafe { MTL4CompilerDescriptor::new() };
            mtl4_device
                .newCompilerWithDescriptor_error(&descriptor)
                .expect("metal4: device could not create an MTL4Compiler")
        };

        let vertex_function_descriptor = unsafe { MTL4LibraryFunctionDescriptor::new() };
        unsafe {
            vertex_function_descriptor.setLibrary(Some(&library));
            vertex_function_descriptor
                .setName(Some(&objc2_foundation::NSString::from_str("quad_vertex")));
        }
        let fragment_function_descriptor = unsafe { MTL4LibraryFunctionDescriptor::new() };
        unsafe {
            fragment_function_descriptor.setLibrary(Some(&library));
            fragment_function_descriptor
                .setName(Some(&objc2_foundation::NSString::from_str("quad_fragment")));
        }

        let pipeline_descriptor = unsafe { MTL4RenderPipelineDescriptor::new() };
        unsafe {
            pipeline_descriptor.setVertexFunctionDescriptor(Some(&vertex_function_descriptor));
            pipeline_descriptor.setFragmentFunctionDescriptor(Some(&fragment_function_descriptor));
            let color_attachment = pipeline_descriptor.colorAttachments().objectAtIndexedSubscript(0);
            color_attachment.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
            color_attachment.setBlendingState(MTL4BlendState::Enabled);
            color_attachment.setRgbBlendOperation(objc2_metal::MTLBlendOperation::Add);
            color_attachment.setAlphaBlendOperation(objc2_metal::MTLBlendOperation::Add);
            color_attachment.setSourceRGBBlendFactor(objc2_metal::MTLBlendFactor::SourceAlpha);
            color_attachment.setSourceAlphaBlendFactor(objc2_metal::MTLBlendFactor::One);
            color_attachment
                .setDestinationRGBBlendFactor(objc2_metal::MTLBlendFactor::OneMinusSourceAlpha);
            color_attachment.setDestinationAlphaBlendFactor(objc2_metal::MTLBlendFactor::One);
        }

        compiler
            .newRenderPipelineStateWithDescriptor_compilerTaskOptions_error(
                &pipeline_descriptor,
                None,
            )
            .expect("metal4: compiler could not build the quad render pipeline state")
    }

    pub fn layer(&self) -> Option<&metal::MetalLayerRef> {
        self.layer.as_ref().map(|l| l.as_ref())
    }

    pub fn layer_ptr(&self) -> *mut metal::CAMetalLayer {
        self.layer
            .as_ref()
            .map(|l| l.as_ptr())
            .unwrap_or(std::ptr::null_mut())
    }

    pub fn sprite_atlas(&self) -> &std::sync::Arc<MetalAtlas> {
        &self.sprite_atlas
    }

    pub fn set_presents_with_transaction(&mut self, presents_with_transaction: bool) {
        if let Some(layer) = &self.layer {
            layer.set_presents_with_transaction(presents_with_transaction);
        }
    }

    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        if let Some(layer) = &self.layer {
            let ns_size = cocoa::foundation::NSSize {
                width: size.width.0 as f64,
                height: size.height.0 as f64,
            };
            unsafe {
                let _: () = msg_send![layer.as_ref(), setDrawableSize: ns_size];
            }
        }
    }

    pub fn update_transparency(&mut self, transparent: bool) {
        self.opaque = !transparent;
        if let Some(layer) = &self.layer {
            layer.set_opaque(!transparent);
        }
    }

    pub fn destroy(&self) {
        // nothing to do
    }

    /// Renders `scene.quads` only — every other primitive kind is silently
    /// skipped. This is the documented scope of the first Metal 4 milestone,
    /// not a bug: see the module doc comment.
    pub fn draw(&mut self, scene: &Scene) {
        let layer = match &self.layer {
            Some(l) => l.clone(),
            None => {
                log::error!("metal4: draw() called on a headless renderer");
                return;
            }
        };

        if !scene.paths.is_empty()
            || !scene.shadows.is_empty()
            || !scene.underlines.is_empty()
            || !scene.monochrome_sprites.is_empty()
            || !scene.polychrome_sprites.is_empty()
            || !scene.surfaces.is_empty()
        {
            log::warn!(
                "metal4: scene has non-quad primitives ({} paths, {} shadows, {} underlines, {} mono, {} poly, {} surfaces) that this milestone renderer does not draw",
                scene.paths.len(),
                scene.shadows.len(),
                scene.underlines.len(),
                scene.monochrome_sprites.len(),
                scene.polychrome_sprites.len(),
                scene.surfaces.len(),
            );
        }

        let viewport_size = layer.drawable_size();
        let viewport_size_px: Size<DevicePixels> = gpui::size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );

        let drawable = match layer.next_drawable() {
            Some(drawable) => drawable,
            None => {
                log::error!("metal4: failed to retrieve next drawable");
                return;
            }
        };

        // Fully serialize frames: wait for the previous frame's GPU work
        // before reusing the sole allocator/command buffer/quads buffer.
        // (A production implementation would triple-buffer these instead,
        // per Apple's guidance — deliberately skipped here, see module doc.)
        let frame_number = self.frame_number.get() + 1;
        self.frame_number.set(frame_number);
        if frame_number > 1 {
            let previous = frame_number - 1;
            let signaled = unsafe { self.shared_event.waitUntilSignaledValue_timeoutMS(previous, 1000) };
            if !signaled {
                log::error!("metal4: timed out waiting for frame {previous} to finish on the GPU");
            }
        }

        unsafe { self.allocator.reset() };
        unsafe {
            self.command_buffer
                .beginCommandBufferWithAllocator(&self.allocator)
        };

        let texture: Retained<ProtocolObject<dyn objc2_metal::MTLTexture>> =
            unsafe { bridge_retain(drawable.texture().as_ptr() as *mut c_void) };

        let render_pass_descriptor = unsafe { MTL4RenderPassDescriptor::new() };
        unsafe {
            let color_attachment = render_pass_descriptor.colorAttachments().objectAtIndexedSubscript(0);
            color_attachment.setTexture(Some(&texture));
            color_attachment.setLoadAction(MTLLoadAction::Clear);
            color_attachment.setStoreAction(MTLStoreAction::Store);
            let alpha = if self.opaque { 1.0 } else { 0.0 };
            color_attachment.setClearColor(MTLClearColor {
                red: 0.0,
                green: 0.0,
                blue: 0.0,
                alpha,
            });
        }

        let encoder = unsafe {
            self.command_buffer
                .renderCommandEncoderWithDescriptor(&render_pass_descriptor)
        }
        .expect("metal4: failed to create a render command encoder");

        unsafe {
            encoder.setViewport(objc2_metal::MTLViewport {
                originX: 0.0,
                originY: 0.0,
                width: i32::from(viewport_size_px.width) as f64,
                height: i32::from(viewport_size_px.height) as f64,
                znear: 0.0,
                zfar: 1.0,
            });
        }

        if !scene.quads.is_empty() {
            self.draw_quads(&scene.quads, viewport_size_px, &encoder);
        }

        unsafe { encoder.endEncoding() };
        unsafe { self.command_buffer.endCommandBuffer() };

        // Present choreography verified against Apple's own sample
        // (Metal4Renderer+Encoding.m's `submitCommandBuffer:toCommandQueue:forView:`):
        // wait for the drawable before commit, commit, signal the drawable
        // after commit, then present — presentation itself is decoupled
        // from the command buffer in Metal 4.
        let mtl_drawable: Retained<ProtocolObject<dyn objc2_metal::MTLDrawable>> =
            unsafe { bridge_retain(drawable.as_ptr() as *mut c_void) };
        unsafe { self.queue.waitForDrawable(&mtl_drawable) };

        let mut command_buffers: [NonNull<ProtocolObject<dyn MTL4CommandBuffer>>; 1] =
            [NonNull::from(&*self.command_buffer)];
        unsafe {
            self.queue
                .commit_count(NonNull::from(&mut command_buffers[0]), 1)
        };

        unsafe { self.queue.signalDrawable(&mtl_drawable) };
        drawable.present();

        unsafe {
            self.queue.signalEvent_value(
                ProtocolObject::from_ref(&*self.shared_event),
                frame_number,
            )
        };
    }

    fn draw_quads(
        &self,
        quads: &[Quad],
        viewport_size: Size<DevicePixels>,
        encoder: &ProtocolObject<dyn MTL4RenderCommandEncoder>,
    ) {
        // Grow the quads buffer (and its residency registration) if needed.
        {
            let mut quads_buffer = self.quads_buffer.borrow_mut();
            if quads.len() > quads_buffer.1 {
                let new_capacity = quads.len().next_power_of_two();
                let new_buffer =
                    Self::new_buffer(&self.mtl4_device, mem::size_of::<Quad>() * new_capacity);
                unsafe {
                    self.residency_set.removeAllocation(quads_buffer.0.as_ref());
                    self.residency_set.addAllocation(new_buffer.as_ref());
                    self.residency_set.commit();
                    self.residency_set.requestResidency();
                }
                *quads_buffer = (new_buffer, new_capacity);
            }
            unsafe {
                std::ptr::copy_nonoverlapping(
                    quads.as_ptr(),
                    quads_buffer.0.contents().as_ptr() as *mut Quad,
                    quads.len(),
                );
            }
        }
        let quads_buffer = self.quads_buffer.borrow();

        unsafe {
            self.viewport_size_buffer
                .contents()
                .cast::<ViewportSize>()
                .as_ptr()
                .write(ViewportSize {
                    width: i32::from(viewport_size.width) as f32,
                    height: i32::from(viewport_size.height) as f32,
                });
        }

        unsafe {
            self.argument_table
                .setAddress_atIndex(self.unit_vertices.gpuAddress(), 0);
            self.argument_table
                .setAddress_atIndex(quads_buffer.0.gpuAddress(), 1);
            self.argument_table
                .setAddress_atIndex(self.viewport_size_buffer.gpuAddress(), 2);
            encoder.setArgumentTable_atStages(
                &self.argument_table,
                MTLRenderStages::Vertex | MTLRenderStages::Fragment,
            );
            encoder.setRenderPipelineState(&self.pipeline_state);
            encoder.drawPrimitives_vertexStart_vertexCount_instanceCount(
                MTLPrimitiveType::Triangle,
                0,
                6,
                quads.len(),
            );
        }
    }
}

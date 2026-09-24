//! A Metal 4 renderer, built up one primitive at a time — now covering
//! every `gpui::Scene` primitive kind that `MetalRenderer` itself actually
//! draws: solid-color quads, underlines, shadows, monochrome sprites,
//! polychrome sprites (the first two that sample a texture — see
//! `draw_monochrome_sprites`), paths (the one genuinely different shape of
//! problem among all of these — a two-pass rasterize-then-composite
//! pipeline, see `draw_paths_to_intermediate`), and surfaces (video frames
//! — `CVPixelBuffer`/`CVMetalTextureCache` interop, see `draw_surfaces`).
//! `Scene` has an eighth field, `subpixel_sprites`, deliberately not ported
//! here either: `MetalRenderer::draw()` itself has
//! `PrimitiveBatch::SubpixelSprites { .. } => unreachable!()`, i.e. gpui
//! doesn't currently produce that primitive in practice at this revision —
//! not a gap relative to the real renderer, just matching its actual scope.
//! See `draw()`'s doc comment for the exact current scope and its one
//! remaining, deliberate simplification (composited paths can double-blend
//! an overlapping region — see `draw_paths_from_intermediate`). Draw order
//! follows `scene.batches()` — the same painter's-algorithm batching
//! `MetalRenderer::draw_primitives_to_texture` uses — so cross-type z-order
//! (e.g. text painted over a quad that's above it in the scene) is correct,
//! not just within-type order.
//!
//! Ported from `metal_renderer.rs`'s device/layer setup (unchanged — Metal 4
//! doesn't touch `CAMetalLayer`/drawable acquisition at all) but the
//! command-submission and resource-binding halves are new: Metal 4 drops
//! automatic resource tracking entirely ("In Metal 4, the framework
//! considers all resources untracked" — Apple's "Understanding the Metal 4
//! core API"), so every buffer/texture the GPU touches must be explicitly
//! placed in an `MTLResidencySet`, and bindings go through an
//! `MTL4ArgumentTable` (a GPU address / `MTLResourceID` table) rather than
//! per-call `setVertexBuffer:`/`setFragmentTexture:`. The shaders themselves
//! are untouched and reused byte-for-byte from the existing
//! `shaders.metallib` — `[[buffer(N)]]` argument-table slots and classic
//! bind-point indices are the same binding namespace from the shader's
//! perspective, only how the CPU side populates slot N differs. Each
//! primitive kind gets its own pipeline (`compile_pipeline`) and its own
//! growable instance buffer — every one of them sub-allocated from a single
//! `instance_heap` (see its field doc comment) rather than standalone
//! `MTLBuffer`s, so growing one is just a fresh sub-allocation with no
//! per-buffer residency-set churn — but all of them reuse the same three
//! argument-table indices (0=unit vertices, 1=primitive data, 2=viewport
//! size) one draw call at a time — `QuadInputIndex`/`UnderlineInputIndex`/
//! etc. all share that exact numeric layout in the shader source.
//!
//! `FRAME_SLOTS`-way triple buffering (Apple's recommended pattern):
//! `FRAME_SLOTS` allocators, command buffers, and copies of every
//! CPU-written-then-GPU-read instance buffer rotate round-robin by frame
//! number, so encoding frame N+1 can start on the CPU while frame N is
//! still executing on the GPU — see `draw()`, `allocators`, and
//! `command_buffers`. Earlier revisions of this renderer used a single
//! allocator/command-buffer/instance-buffer set, fully serialized (each
//! frame blocked on the GPU finishing before the next began, via an
//! `MTLSharedEvent`) — correctness-focused, deliberately non-pipelined,
//! and superseded by the triple-buffered version once its extra
//! bookkeeping (see `write_full`'s doc comment for the specific bug that
//! bookkeeping exists to prevent) was worth taking on.
//!
//! Verified against Apple's own "Drawing a Triangle with Metal 4" sample
//! (`Metal4Renderer.m` / `+Setup.m` / `+Compilation.m` / `+Encoding.m`) for
//! the parts that aren't otherwise documented in the Metal 4 API reference:
//! command-buffer reuse via `beginCommandBufferWithAllocator:` after commit,
//! queue-level (not command-buffer-level) residency via `addResidencySet:`,
//! and the drawable present choreography (`waitForDrawable:` before commit,
//! `signalDrawable:` after commit, then `[drawable present]`). This
//! renderer diverges from the sample in one place — see
//! `command_buffers`'s doc comment for why each frame slot gets its own
//! command buffer object rather than sharing the sample's single one.

use crate::metal_atlas::MetalAtlas;
use crate::metal_renderer::{PathRasterizationVertex, PathSprite, SurfaceBounds};
#[cfg(target_os = "macos")]
use core_foundation::base::TCFType;
#[cfg(target_os = "macos")]
use core_video::{
    metal_texture::CVMetalTextureGetTexture, metal_texture_cache::CVMetalTextureCache,
    pixel_buffer::kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
};
use foreign_types::{ForeignType, ForeignTypeRef};
use gpui::{
    DevicePixels, MonochromeSprite, PaintSurface, Path, PolychromeSprite, PrimitiveBatch, Quad,
    ScaledPixels, Scene, Size,
};
// Only referenced by name in the `#[cfg(test)]` fixtures further down this
// file — `draw_shadows`/`draw_underlines` themselves take a `Range<usize>`
// into an already-written buffer, not a `&[Shadow]`/`&[Underline]` slice.
#[cfg(test)]
use gpui::{Shadow, Underline};
use objc::{msg_send, sel, sel_impl};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4BlendState, MTL4CommandAllocator,
    MTL4CommandBuffer, MTL4CommandEncoder, MTL4CommandQueue, MTL4Compiler, MTL4CompilerDescriptor,
    MTL4LibraryFunctionDescriptor, MTL4RenderCommandEncoder, MTL4RenderPassDescriptor,
    MTL4RenderPipelineDescriptor, MTLBuffer, MTLClearColor, MTLDevice, MTLHeap, MTLHeapDescriptor,
    MTLLoadAction, MTLPixelFormat, MTLPrimitiveType, MTLRenderPipelineState, MTLRenderStages,
    MTLResidencySet, MTLResidencySetDescriptor, MTLResourceOptions, MTLSharedEvent, MTLStorageMode,
    MTLStoreAction, MTLTexture,
};
use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use std::mem;
use std::ops::Range;
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

/// Number of in-flight frame slots — Apple's recommended triple buffering,
/// see the module doc comment. Each slot owns its own `MTL4CommandAllocator`
/// and `MTL4CommandBuffer` pair, and its own copy of every
/// CPU-written-then-GPU-read instance buffer (`quads_buffer`,
/// `viewport_size_buffer`, etc. — see `draw()`'s doc comment for exactly
/// which fields and why). `argument_table` is the one exception — *not*
/// per-slot, see its own doc comment for why sharing it across overlapping
/// frames is safe.
const FRAME_SLOTS: usize = 3;

const MAX_QUADS_PER_FRAME_INITIAL: usize = 256;
/// Same reasoning as `metal_renderer.rs`'s own `PATH_SAMPLE_COUNT` (not
/// reused directly — that constant is private to that file, and this is a
/// one-line, zero-risk duplication rather than widening its visibility):
/// 4x MSAA, which every device supports.
/// https://developer.apple.com/documentation/metal/mtldevice/1433355-supportstexturesamplecount
const PATH_SAMPLE_COUNT: u32 = 4;
/// Initial capacity for the flattened path-vertex buffer — each path
/// contributes a variable number of triangles (3 vertices each), so this is
/// sized as "a few dozen small paths' worth" rather than mirroring
/// `MAX_QUADS_PER_FRAME_INITIAL` 1:1 the way the other instance buffers do.
const MAX_PATH_VERTICES_INITIAL: usize = 3 * 256;
/// Initial slot count for `atlas_size_buffer` — like every other instance
/// buffer's `_INITIAL` constant, just a starting point; `draw_batches`
/// grows it (once per frame, before the batch loop — see `ensure_capacity`)
/// to fit however many distinct sprite-atlas sizes this *particular* frame
/// actually needs, with no fixed ceiling.
const ATLAS_SIZE_SLOTS_INITIAL: usize = 8;
/// Starting size of `instance_heap` — same 2 MB default `MetalRenderer`'s own
/// `InstanceBufferPool` uses (see `metal_renderer.rs`), comfortably larger
/// than every `_INITIAL` buffer size above summed together (a few hundred
/// KB at most), leaving headroom to grow several times over before
/// `Self::new_buffer`'s fallback path (a standalone, individually
/// residency-tracked buffer, for when the heap is genuinely out of room)
/// ever has to engage in practice.
const INSTANCE_HEAP_INITIAL_SIZE: usize = 2 * 1024 * 1024;

/// Bit-for-bit layout of the shader's `Size_DevicePixels` (cbindgen-generated
/// from `gpui::Size<DevicePixels>`, itself two `i32`s) — **not** `f32`. Using
/// the wrong field type here was an actual bug caught by the headless quad
/// test: reinterpreting a viewport-size `f32` bit pattern as `int32_t`
/// produces a garbage viewport size, which pushes every quad's clip-space
/// position outside the -1..1 range — i.e. a fully black frame.
#[repr(C)]
struct ViewportSize {
    width: i32,
    height: i32,
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
    /// `FRAME_SLOTS`-way round-robin, paired one-to-one with `allocators`
    /// (same slot index, same lifecycle). `MTL4CommandBuffer::
    /// beginCommandBufferWithAllocator:`'s own doc comment only documents
    /// the *allocator* as safe to reuse right after `endCommandBuffer`
    /// ("You can safely reuse command allocators after ending the command
    /// buffer using it by calling endCommandBuffer") — it says nothing
    /// about whether the *command buffer object itself* may safely be
    /// re-begun before its own prior submission has finished executing on
    /// the GPU. Rather than assume that's fine, each slot gets its own
    /// command buffer object and only touches it again once `draw()` has
    /// already confirmed (via `shared_event`) that slot's last use
    /// completed — the same wait `allocators`' `reset()` needs anyway, so
    /// this costs nothing extra and removes the ambiguity entirely. (The
    /// original single-command-buffer version of this renderer relied on
    /// the *allocator*-reuse guarantee alone, safely, only because it also
    /// fully serialized every frame behind a GPU-completion wait — a
    /// stronger condition than the allocator's contract actually needs.)
    command_buffers: [Retained<ProtocolObject<dyn MTL4CommandBuffer>>; FRAME_SLOTS],
    /// `FRAME_SLOTS`-way round-robin, one allocator per in-flight frame.
    /// `frame_number % FRAME_SLOTS` (via `current_slot`) picks which one
    /// `draw()`/`render_scene_to_image()` binds to that same-indexed
    /// `command_buffers` slot this frame. `MTL4CommandAllocator::reset()`'s
    /// own doc comment: "You are responsible to ensure that all command
    /// buffers with memory originating from this allocator instance are
    /// complete before [resetting] it" — `draw()` only resets a slot's
    /// allocator after confirming (via `shared_event`) that the *last*
    /// frame which used that slot (`frame_number - FRAME_SLOTS` frames ago)
    /// has finished on the GPU. With one allocator this meant waiting for
    /// the *previous* frame every time (full CPU/GPU serialization, the
    /// simplification this renderer shipped with initially); with
    /// `FRAME_SLOTS` of them, encoding frame N+1 can start while frame N is
    /// still executing, since they use different allocators — the actual
    /// point of triple buffering.
    allocators: [Retained<ProtocolObject<dyn MTL4CommandAllocator>>; FRAME_SLOTS],
    /// Shared across every slot/frame, unlike the instance buffers below —
    /// safe because `setArgumentTable_atStages` captures the table's
    /// *current* bindings into that specific draw call at encode time (not
    /// a live pointer the GPU re-reads later): every `draw_*` method here
    /// already calls `setAddress_atIndex` + `setArgumentTable_atStages`
    /// fresh, immediately before its own `drawPrimitives`, and did so
    /// safely even when multiple draw calls of *different* primitive kinds
    /// shared this same table within one frame, before `FRAME_SLOTS` even
    /// existed — the same reasoning extends unchanged across frames.
    argument_table: Retained<ProtocolObject<dyn MTL4ArgumentTable>>,
    /// Long-lived residency set for everything except the drawable itself
    /// (the drawable's own residency is `CAMetalLayer.residencySet`, added
    /// to the queue separately in `new` — see the doc comment up top).
    residency_set: Retained<ProtocolObject<dyn MTLResidencySet>>,
    /// Backs every renderer-owned `MTLBuffer` (`unit_vertices`,
    /// `viewport_size_buffer`, and every field below sub-allocates from
    /// this via `Self::new_buffer`) — added to `residency_set` exactly
    /// once, here, in `new_internal`. `MTLHeap: MTLAllocation`, so
    /// everything sub-allocated from a resident heap is itself
    /// automatically resident; no buffer growth anywhere in this renderer
    /// needs its own `addAllocation`/`removeAllocation`/`commit`/
    /// `requestResidency` call any more, unlike before this field existed.
    /// `Self::new_buffer` falls back to a standalone, individually-tracked
    /// buffer on the rare frame that exhausts it (see
    /// `INSTANCE_HEAP_INITIAL_SIZE`) — this renderer still has no hard
    /// buffer-size ceiling, the heap just makes the common case cheaper.
    instance_heap: Retained<ProtocolObject<dyn MTLHeap>>,
    quad_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    underline_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    shadow_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    monochrome_sprite_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    polychrome_sprite_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    path_rasterization_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    path_sprite_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    surface_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    /// Vends a `CVMetalTexture` (Y and CbCr planes, separately) from each
    /// `PaintSurface`'s `CVPixelBuffer` every `draw_surfaces` call — a real
    /// GPU-texture cache, not something this renderer manages itself; kept
    /// alive for the renderer's whole lifetime like `MetalRenderer`'s own.
    #[cfg(target_os = "macos")]
    core_video_texture_cache: CVMetalTextureCache,
    /// Never written after construction (the constant unit quad), so —
    /// unlike every buffer below — it needs no per-slot copy: there's
    /// nothing for an overlapping frame to race against.
    unit_vertices: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// One copy per frame slot (see `current_slot`) — unlike
    /// `argument_table`, this buffer's *contents* are read live by the GPU
    /// (a raw pointer dereference in the shader), not captured at encode
    /// time, and unlike `unit_vertices` its value genuinely can change
    /// frame to frame (a resize). With triple buffering, frame N+1 can
    /// start encoding — and writing a new viewport size — while frame N is
    /// still executing on the GPU; a single shared buffer would let frame
    /// N+1's write reach frame N's still-in-flight draw calls before they
    /// run, corrupting frame N's viewport. See `draw()`'s doc comment for
    /// the general rule this and every buffer below follow.
    viewport_size_buffer: [Retained<ProtocolObject<dyn MTLBuffer>>; FRAME_SLOTS],
    /// One `ViewportSize`-sized slot per sprite-batch draw call this frame
    /// (see `atlas_size_cursor`), grown like every other instance buffer —
    /// unlike `viewport_size_buffer`'s old single-buffer self, which used to
    /// be safe to blindly overwrite because every draw call in a frame
    /// wrote the *same* value to it, a sprite batch's atlas size genuinely
    /// differs from another sprite batch's in the same frame whenever they
    /// sample different atlas textures (`scene.batches()` splits
    /// `MonochromeSprites`/`PolychromeSprites` by `texture_id` exactly when
    /// that happens). Per-slot for the same cross-frame reason
    /// `viewport_size_buffer` is. Also used, unslotted (always its element
    /// 0 of the current frame slot), by `draw_surfaces` for a video frame's
    /// texture size — a separate, pre-existing, narrower simplification
    /// (`draw_surfaces` writes one `SurfaceBounds`/one atlas size per
    /// surface in an internal loop, so 2+ `PaintSurface`s in one frame — or
    /// a frame with both surfaces and sprites — can race the same way; see
    /// `draw_surfaces`'s own doc comment. Not fixed here: this app never
    /// paints a `PaintSurface`).
    atlas_size_buffer: [RefCell<(Retained<ProtocolObject<dyn MTLBuffer>>, usize)>; FRAME_SLOTS],
    /// Next free element index into the *current slot's*
    /// `atlas_size_buffer` — reset to 0 at the top of every `draw_batches`
    /// call, bumped by one per `draw_monochrome_sprites`/
    /// `draw_polychrome_sprites` call. Not itself per-slot: it's transient,
    /// scoped to "this frame's encoding" and reset unconditionally
    /// regardless of which slot that encoding targets.
    atlas_size_cursor: Cell<usize>,
    /// One copy per frame slot — reallocated (sub-allocated fresh from
    /// `instance_heap` — see its field doc comment — no residency-set
    /// bookkeeping needed) whenever a frame needs more quads than that
    /// slot's current capacity. See `draw()`'s doc comment for why every
    /// instance buffer needs `FRAME_SLOTS` copies once frame N+1's CPU
    /// encoding can overlap frame N's GPU execution.
    quads_buffer: [RefCell<(Retained<ProtocolObject<dyn MTLBuffer>>, usize)>; FRAME_SLOTS],
    /// Same growth strategy as `quads_buffer`, for `gpui::Underline`s.
    underlines_buffer: [RefCell<(Retained<ProtocolObject<dyn MTLBuffer>>, usize)>; FRAME_SLOTS],
    /// Same growth strategy as `quads_buffer`, for `gpui::Shadow`s.
    shadows_buffer: [RefCell<(Retained<ProtocolObject<dyn MTLBuffer>>, usize)>; FRAME_SLOTS],
    /// Same growth strategy as `quads_buffer`, for `gpui::MonochromeSprite`s.
    monochrome_sprites_buffer:
        [RefCell<(Retained<ProtocolObject<dyn MTLBuffer>>, usize)>; FRAME_SLOTS],
    /// Same growth strategy as `quads_buffer`, for `gpui::PolychromeSprite`s.
    polychrome_sprites_buffer:
        [RefCell<(Retained<ProtocolObject<dyn MTLBuffer>>, usize)>; FRAME_SLOTS],
    /// One copy per frame slot. Flattened `PathRasterizationVertex`es — not
    /// for one path batch, but bump-allocated across *every* `Paths` batch
    /// in the frame (see `path_vertices_cursor`): `draw_batches` pre-grows
    /// the current slot's buffer to the whole scene's total vertex count
    /// before the batch loop starts (from `scene.paths`, summed across
    /// every path, not just one batch), so a second `Paths` batch's
    /// rasterization never has to reallocate — which would free the buffer
    /// a first batch's already-encoded-but-not-yet-GPU-executed
    /// rasterization draw this same frame still points at (see
    /// `draw_batches`'s doc comment). Not instanced like the other buffers
    /// (each vertex is one corner of one triangle, drawn with a plain,
    /// non-instanced `drawPrimitives`, offset via `vertexStart` instead of
    /// `baseInstance`).
    path_vertices_buffer: [RefCell<(Retained<ProtocolObject<dyn MTLBuffer>>, usize)>; FRAME_SLOTS],
    /// Next free element index into the current slot's `path_vertices_buffer`
    /// — reset to 0 at the top of every `draw_batches` call, advanced by
    /// each `Paths` batch's vertex count. Not per-slot itself, same
    /// reasoning as `atlas_size_cursor`.
    path_vertices_cursor: Cell<usize>,
    /// One copy per frame slot. One `PathSprite` (just a `Bounds`) per
    /// path, used only by the compositing pass — see
    /// `draw_paths_from_intermediate`. Same bump-allocated-across-the-
    /// whole-frame treatment as `path_vertices_buffer`, pre-grown to
    /// `scene.paths.len()`.
    path_sprites_buffer: [RefCell<(Retained<ProtocolObject<dyn MTLBuffer>>, usize)>; FRAME_SLOTS],
    /// Next free element index into the current slot's `path_sprites_buffer`
    /// — reset to 0 at the top of every `draw_batches` call, advanced by
    /// each `Paths` batch's path count. Not per-slot itself, same reasoning
    /// as `atlas_size_cursor`.
    path_sprites_cursor: Cell<usize>,
    /// Full-viewport-sized, rebuilt on resize (`draw_paths_to_intermediate`
    /// checks the size itself, lazily, only when there's a path to
    /// rasterize — unlike `MetalRenderer`, which always keeps this current
    /// via an explicit `update_path_intermediate_textures` call). Legacy
    /// `metal` crate textures, like every other renderer-owned texture in
    /// this file, bridged to objc2 only where the MTL4 APIs need it.
    path_intermediate_texture: RefCell<Option<metal::Texture>>,
    /// `None` when `path_sample_count <= 1` (never happens today —
    /// `PATH_SAMPLE_COUNT` is a fixed constant — but mirrors
    /// `MetalRenderer`'s structure in case that ever becomes configurable).
    path_intermediate_msaa_texture: RefCell<Option<metal::Texture>>,
    /// The `(width, height)` the two textures above were last built for.
    path_intermediate_size: Cell<Option<(i32, i32)>>,
    /// Single-slot, unlike every other instance buffer: `draw_surfaces`
    /// draws one `PaintSurface` at a time in a loop (each has its own pair
    /// of Y/CbCr textures fetched fresh from the video frame, so there's no
    /// batching win to instancing them together the way quads/sprites are),
    /// rewriting this buffer's one `SurfaceBounds` before each draw.
    surfaces_buffer: RefCell<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// The Y/CbCr `CVMetalTexture`-backed textures added to `residency_set`
    /// by the *previous* `draw_surfaces` call — unlike atlas/path
    /// intermediate textures, these are brand new objects every single
    /// call (a fresh video frame each time), so there's nothing to cache by
    /// identity; this exists purely so `draw_surfaces` can remove last
    /// call's entries before adding this call's, instead of letting the
    /// residency set accumulate a new pair of stale texture references
    /// forever.
    surface_textures_in_residency: RefCell<Vec<Retained<ProtocolObject<dyn MTLTexture>>>>,
    /// The most recent atlas texture added to `residency_set` by
    /// `draw_monochrome_sprites`/`draw_polychrome_sprites` — re-added only
    /// when a draw call needs a *different* texture than last time, since
    /// `MetalAtlas` allocates its own textures outside this renderer's
    /// control and Metal 4 has no automatic residency to fall back on if
    /// one is missed. Shared between both sprite kinds for simplicity: a
    /// frame that alternates between monochrome and polychrome sprites
    /// re-adds each texture more often than strictly necessary, which is
    /// wasted work but not a correctness problem (residency-set membership
    /// is additive — a redundant `addAllocation` of an already-resident
    /// texture is harmless).
    resident_atlas_texture: Cell<Option<gpui::AtlasTextureId>>,
    /// Signalled with `frame_number` after each commit. `draw()`/
    /// `render_scene_to_image()` wait on it — not for the *previous* frame
    /// any more (that was the single-allocator simplification), but for
    /// `frame_number - FRAME_SLOTS`: the last frame that used the *same*
    /// allocator/instance-buffer slot this frame is about to reuse. See
    /// `allocators`'s doc comment.
    shared_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    frame_number: Cell<u64>,
    /// `(frame_number - 1) % FRAME_SLOTS` (frame numbers start at 1) —
    /// computed once per frame, right after `frame_number` is bumped, and
    /// read by every method that indexes a per-slot field (`quads_buffer`
    /// and friends) so the frame's own slot doesn't have to be threaded
    /// through every one of their call signatures.
    current_slot: Cell<usize>,
}

impl Metal4Renderer {
    #[cfg(target_os = "macos")]
    pub fn new(transparent: bool) -> Self {
        let device = Self::create_device();

        let layer = metal::MetalLayer::new();
        Self::configure_layer(&layer, &device, transparent);
        unsafe {
            let _: () = msg_send![&*layer, setAutoresizingMask: 18_u32];
        }

        Self::new_internal(device, Some(layer), !transparent)
    }

    /// Creates a renderer for a `CAMetalLayer` owned by a platform view (iOS: the view's own
    /// backing layer).
    ///
    /// # Safety
    ///
    /// `layer` must point to a live `CAMetalLayer` and only be used from the thread its owning
    /// view may be accessed on.
    #[cfg(target_os = "ios")]
    pub unsafe fn from_layer(layer: *mut metal::CAMetalLayer, transparent: bool) -> Self {
        let device = Self::create_device();
        let retained_layer: *mut metal::CAMetalLayer = unsafe { msg_send![layer, retain] };
        let layer = unsafe { metal::MetalLayer::from_ptr(retained_layer) };
        Self::configure_layer(&layer, &device, transparent);
        Self::new_internal(device, Some(layer), !transparent)
    }

    fn configure_layer(layer: &metal::MetalLayerRef, device: &metal::DeviceRef, transparent: bool) {
        layer.set_device(device);
        layer.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        layer.set_opaque(!transparent);
        layer.set_maximum_drawable_count(3);
        unsafe {
            let _: () = msg_send![layer, setAllowsNextDrawableTimeout: objc::runtime::NO];
            let _: () = msg_send![layer, setNeedsDisplayOnBoundsChange: objc::runtime::YES];
        }
    }

    /// Creates a Metal4Renderer with no `CAMetalLayer`, for offscreen
    /// rendering via `render_scene_to_image` — see that method and
    /// `MetalRenderer::new_headless` for the equivalent Metal 3 path.
    #[cfg(any(test, feature = "test-support"))]
    pub fn new_headless() -> Self {
        let device = Self::create_device();
        Self::new_internal(device, None, true)
    }

    #[cfg(target_os = "ios")]
    fn create_device() -> metal::Device {
        metal::Device::system_default().unwrap_or_else(|| {
            log::error!("metal4: unable to access a compatible graphics device");
            std::process::exit(1);
        })
    }

    #[cfg(target_os = "macos")]
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

    fn new_internal(device: metal::Device, layer: Option<metal::MetalLayer>, opaque: bool) -> Self {
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
        let command_buffers: [Retained<ProtocolObject<dyn MTL4CommandBuffer>>; FRAME_SLOTS] =
            std::array::from_fn(|_| {
                mtl4_device
                    .newCommandBuffer()
                    .expect("metal4: device could not create an MTL4CommandBuffer")
            });
        let allocators: [Retained<ProtocolObject<dyn MTL4CommandAllocator>>; FRAME_SLOTS] =
            std::array::from_fn(|_| {
                mtl4_device
                    .newCommandAllocator()
                    .expect("metal4: device could not create an MTL4CommandAllocator")
            });

        let argument_table = {
            let descriptor = MTL4ArgumentTableDescriptor::new();
            // Buffer indices 0..=3 (unit vertices, primitive data, viewport
            // size, atlas texture size — see `SpriteInputIndex` in
            // shaders.metal) and texture index 4 (the atlas texture
            // itself). Buffers and textures are separate lists within one
            // argument table, same as classic Metal's separate
            // setVertexBuffer:/setVertexTexture: namespaces.
            descriptor.setMaxBufferBindCount(4);
            descriptor.setMaxTextureBindCount(5);
            mtl4_device
                .newArgumentTableWithDescriptor_error(&descriptor)
                .expect("metal4: device could not create an MTL4ArgumentTable")
        };

        let residency_set = {
            let descriptor = MTLResidencySetDescriptor::new();
            mtl4_device
                .newResidencySetWithDescriptor_error(&descriptor)
                .expect("metal4: device could not create an MTLResidencySet")
        };

        // Every renderer-owned `MTLBuffer` sub-allocates from this — see
        // `instance_heap`'s field doc comment. `StorageModeShared` since
        // this renderer only ever writes buffer contents via a CPU pointer
        // (`contents()`), never a blit; every `Self::new_buffer` call below
        // matches this with its own `options` argument, as heap-buffer
        // creation requires (a mismatched storage/cache mode is a
        // programmer error `newBufferWithLength:options:` documents as
        // producing `nil`, not UB — but silently, so it's still on us to
        // keep them in sync).
        let instance_heap = {
            let descriptor = MTLHeapDescriptor::new();
            descriptor.setSize(INSTANCE_HEAP_INITIAL_SIZE);
            descriptor.setStorageMode(MTLStorageMode::Shared);
            mtl4_device
                .newHeapWithDescriptor(&descriptor)
                .expect("metal4: device could not create an MTLHeap")
        };
        {
            residency_set.addAllocation(instance_heap.as_ref());
            residency_set.commit();
            residency_set.requestResidency();
        }

        let shared_event = mtl4_device
            .newSharedEvent()
            .expect("metal4: device could not create an MTLSharedEvent");
        shared_event.setSignaledValue(0);

        // The unit quad (two triangles covering [0,1]x[0,1]) — identical
        // data/layout to MetalRenderer's `unit_vertices`, just written
        // through objc2-metal instead of the legacy `metal` crate.
        let unit_vertices_data: [[f32; 2]; 6] =
            [[0., 0.], [1., 0.], [0., 1.], [0., 1.], [1., 0.], [1., 1.]];
        let unit_vertices = Self::new_buffer_with_data(
            &mtl4_device,
            &instance_heap,
            &residency_set,
            unit_vertices_data.as_ptr() as *const c_void,
            mem::size_of_val(&unit_vertices_data),
        );

        let viewport_size_buffer: [Retained<ProtocolObject<dyn MTLBuffer>>; FRAME_SLOTS] =
            std::array::from_fn(|_| {
                Self::new_buffer(
                    &mtl4_device,
                    &instance_heap,
                    &residency_set,
                    mem::size_of::<ViewportSize>(),
                )
            });

        let quads_buffer: [RefCell<(Retained<ProtocolObject<dyn MTLBuffer>>, usize)>; FRAME_SLOTS] =
            std::array::from_fn(|_| {
                RefCell::new((
                    Self::new_buffer(
                        &mtl4_device,
                        &instance_heap,
                        &residency_set,
                        mem::size_of::<Quad>() * MAX_QUADS_PER_FRAME_INITIAL,
                    ),
                    MAX_QUADS_PER_FRAME_INITIAL,
                ))
            });

        queue.addResidencySet(&residency_set);

        // The drawable's own texture needs to be resident too, via the
        // layer's own residency set — a real, separate `MTLResidencySet`
        // Metal 4-aware `CAMetalLayer`s expose (confirmed via Apple's
        // "drawing-a-triangle-with-metal-4" sample, which adds it to the
        // queue the exact same way). The legacy `metal` crate has no
        // binding for this new property, so it's fetched via a raw
        // `objc` message send and bridged the same way as the device above.
        if let Some(layer) = &layer {
            let layer_residency_set_ptr: *mut c_void = unsafe { msg_send![&**layer, residencySet] };
            if !layer_residency_set_ptr.is_null() {
                let layer_residency_set: Retained<ProtocolObject<dyn MTLResidencySet>> =
                    unsafe { bridge_retain_autoreleased(layer_residency_set_ptr) };
                queue.addResidencySet(&layer_residency_set);
            } else {
                log::warn!(
                    "metal4: CAMetalLayer.residencySet was nil — the drawable's own texture may not be resident"
                );
            }
        }

        let underlines_buffer: [RefCell<(Retained<ProtocolObject<dyn MTLBuffer>>, usize)>;
            FRAME_SLOTS] = std::array::from_fn(|_| {
            RefCell::new((
                Self::new_buffer(
                    &mtl4_device,
                    &instance_heap,
                    &residency_set,
                    mem::size_of::<gpui::Underline>() * MAX_QUADS_PER_FRAME_INITIAL,
                ),
                MAX_QUADS_PER_FRAME_INITIAL,
            ))
        });

        let shadows_buffer: [RefCell<(Retained<ProtocolObject<dyn MTLBuffer>>, usize)>;
            FRAME_SLOTS] = std::array::from_fn(|_| {
            RefCell::new((
                Self::new_buffer(
                    &mtl4_device,
                    &instance_heap,
                    &residency_set,
                    mem::size_of::<gpui::Shadow>() * MAX_QUADS_PER_FRAME_INITIAL,
                ),
                MAX_QUADS_PER_FRAME_INITIAL,
            ))
        });

        let monochrome_sprites_buffer: [RefCell<(Retained<ProtocolObject<dyn MTLBuffer>>, usize)>;
            FRAME_SLOTS] = std::array::from_fn(|_| {
            RefCell::new((
                Self::new_buffer(
                    &mtl4_device,
                    &instance_heap,
                    &residency_set,
                    mem::size_of::<MonochromeSprite>() * MAX_QUADS_PER_FRAME_INITIAL,
                ),
                MAX_QUADS_PER_FRAME_INITIAL,
            ))
        });

        let polychrome_sprites_buffer: [RefCell<(Retained<ProtocolObject<dyn MTLBuffer>>, usize)>;
            FRAME_SLOTS] = std::array::from_fn(|_| {
            RefCell::new((
                Self::new_buffer(
                    &mtl4_device,
                    &instance_heap,
                    &residency_set,
                    mem::size_of::<PolychromeSprite>() * MAX_QUADS_PER_FRAME_INITIAL,
                ),
                MAX_QUADS_PER_FRAME_INITIAL,
            ))
        });

        let path_vertices_buffer: [RefCell<(Retained<ProtocolObject<dyn MTLBuffer>>, usize)>;
            FRAME_SLOTS] = std::array::from_fn(|_| {
            RefCell::new((
                Self::new_buffer(
                    &mtl4_device,
                    &instance_heap,
                    &residency_set,
                    mem::size_of::<PathRasterizationVertex>() * MAX_PATH_VERTICES_INITIAL,
                ),
                MAX_PATH_VERTICES_INITIAL,
            ))
        });

        let path_sprites_buffer: [RefCell<(Retained<ProtocolObject<dyn MTLBuffer>>, usize)>;
            FRAME_SLOTS] = std::array::from_fn(|_| {
            RefCell::new((
                Self::new_buffer(
                    &mtl4_device,
                    &instance_heap,
                    &residency_set,
                    mem::size_of::<PathSprite>() * MAX_QUADS_PER_FRAME_INITIAL,
                ),
                MAX_QUADS_PER_FRAME_INITIAL,
            ))
        });

        // Not per-slot — see its field doc comment (the pre-existing,
        // out-of-scope internal-loop simplification `draw_surfaces` already
        // has makes a second buffering axis here moot in practice).
        let surfaces_buffer = Self::new_buffer(
            &mtl4_device,
            &instance_heap,
            &residency_set,
            mem::size_of::<SurfaceBounds>(),
        );
        #[cfg(target_os = "macos")]
        let core_video_texture_cache = CVMetalTextureCache::new(None, device.clone(), None)
            .expect("metal4: could not create a CVMetalTextureCache");

        let atlas_size_buffer: [RefCell<(Retained<ProtocolObject<dyn MTLBuffer>>, usize)>;
            FRAME_SLOTS] = std::array::from_fn(|_| {
            RefCell::new((
                Self::new_buffer(
                    &mtl4_device,
                    &instance_heap,
                    &residency_set,
                    mem::size_of::<ViewportSize>() * ATLAS_SIZE_SLOTS_INITIAL,
                ),
                ATLAS_SIZE_SLOTS_INITIAL,
            ))
        });

        let library = Self::load_shader_library(&device);
        let compiler = {
            let descriptor = MTL4CompilerDescriptor::new();
            mtl4_device
                .newCompilerWithDescriptor_error(&descriptor)
                .expect("metal4: device could not create an MTL4Compiler")
        };
        let quad_pipeline_state =
            Self::compile_pipeline(&compiler, &library, "quad_vertex", "quad_fragment");
        let underline_pipeline_state = Self::compile_pipeline(
            &compiler,
            &library,
            "underline_vertex",
            "underline_fragment",
        );
        let shadow_pipeline_state =
            Self::compile_pipeline(&compiler, &library, "shadow_vertex", "shadow_fragment");
        let monochrome_sprite_pipeline_state = Self::compile_pipeline(
            &compiler,
            &library,
            "monochrome_sprite_vertex",
            "monochrome_sprite_fragment",
        );
        let polychrome_sprite_pipeline_state = Self::compile_pipeline(
            &compiler,
            &library,
            "polychrome_sprite_vertex",
            "polychrome_sprite_fragment",
        );
        let path_rasterization_pipeline_state =
            Self::compile_path_rasterization_pipeline(&compiler, &library);
        let path_sprite_pipeline_state = Self::compile_path_sprite_pipeline(&compiler, &library);
        let surface_pipeline_state =
            Self::compile_pipeline(&compiler, &library, "surface_vertex", "surface_fragment");

        Self {
            device,
            layer,
            opaque,
            sprite_atlas,
            mtl4_device,
            queue,
            command_buffers,
            allocators,
            argument_table,
            residency_set,
            instance_heap,
            quad_pipeline_state,
            underline_pipeline_state,
            shadow_pipeline_state,
            monochrome_sprite_pipeline_state,
            polychrome_sprite_pipeline_state,
            path_rasterization_pipeline_state,
            path_sprite_pipeline_state,
            surface_pipeline_state,
            #[cfg(target_os = "macos")]
            core_video_texture_cache,
            unit_vertices,
            viewport_size_buffer,
            atlas_size_buffer,
            atlas_size_cursor: Cell::new(0),
            quads_buffer,
            underlines_buffer,
            shadows_buffer,
            monochrome_sprites_buffer,
            polychrome_sprites_buffer,
            path_vertices_buffer,
            path_vertices_cursor: Cell::new(0),
            path_sprites_buffer,
            path_sprites_cursor: Cell::new(0),
            path_intermediate_texture: RefCell::new(None),
            path_intermediate_msaa_texture: RefCell::new(None),
            path_intermediate_size: Cell::new(None),
            surfaces_buffer: RefCell::new(surfaces_buffer),
            surface_textures_in_residency: RefCell::new(Vec::new()),
            resident_atlas_texture: Cell::new(None),
            shared_event,
            frame_number: Cell::new(0),
            current_slot: Cell::new(0),
        }
    }

    /// Sub-allocates a `length`-byte buffer from `heap` — the normal path
    /// for every renderer-owned buffer, resident automatically because
    /// `heap` itself is `residency_set`'s only member covering these (see
    /// `instance_heap`'s field doc comment). Only on the genuinely rare
    /// frame where `heap` has no room left (`newBufferWithLength:options:`
    /// returns `nil` — never a hard error) does this fall back to a
    /// standalone, individually-tracked buffer, so a pathological scene
    /// degrades rather than failing outright; logged, since that fallback
    /// buffer's `addAllocation` is a real, permanent residency-set entry a
    /// caller has to `removeAllocation` itself when the buffer is replaced
    /// (`write_full`/`ensure_capacity` already do, unconditionally, since
    /// removing a non-member is a documented-safe no-op).
    fn new_buffer(
        device: &ProtocolObject<dyn MTLDevice>,
        heap: &ProtocolObject<dyn MTLHeap>,
        residency_set: &ProtocolObject<dyn MTLResidencySet>,
        length: usize,
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        if let Some(buffer) =
            heap.newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)
        {
            return buffer;
        }
        log::warn!(
            "metal4: instance heap exhausted ({length} more bytes needed) — falling back to a standalone, individually resident buffer"
        );
        let buffer =
            { device.newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared) }
                .expect("metal4: device could not allocate a fallback MTLBuffer");
        {
            residency_set.addAllocation(buffer.as_ref());
            residency_set.commit();
            residency_set.requestResidency();
        }
        buffer
    }

    fn new_buffer_with_data(
        device: &ProtocolObject<dyn MTLDevice>,
        heap: &ProtocolObject<dyn MTLHeap>,
        residency_set: &ProtocolObject<dyn MTLResidencySet>,
        bytes: *const c_void,
        length: usize,
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let buffer = Self::new_buffer(device, heap, residency_set, length);
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes as *const u8,
                buffer.contents().as_ptr() as *mut u8,
                length,
            );
        }
        buffer
    }

    /// Loads the exact same, unmodified shaders.metallib MetalRenderer
    /// builds from — the shader functions' `[[buffer(N)]]` argument-table
    /// slots are the same binding namespace regardless of whether the CPU
    /// side populated them via classic setVertexBuffer:/setFragmentBuffer:
    /// or an MTL4ArgumentTable. Mirrors metal_renderer.rs's own
    /// runtime_shaders/precompiled split (this build enables the
    /// `runtime_shaders` feature, so only that branch is actually exercised
    /// here, but both are kept for parity).
    fn load_shader_library(
        legacy_device: &metal::Device,
    ) -> Retained<ProtocolObject<dyn objc2_metal::MTLLibrary>> {
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
        unsafe { bridge_retain(legacy_library.as_ptr() as *mut c_void) }
    }

    fn compile_pipeline(
        compiler: &ProtocolObject<dyn MTL4Compiler>,
        library: &ProtocolObject<dyn objc2_metal::MTLLibrary>,
        vertex_fn_name: &str,
        fragment_fn_name: &str,
    ) -> Retained<ProtocolObject<dyn MTLRenderPipelineState>> {
        let vertex_function_descriptor = MTL4LibraryFunctionDescriptor::new();
        {
            vertex_function_descriptor.setLibrary(Some(library));
            vertex_function_descriptor
                .setName(Some(&objc2_foundation::NSString::from_str(vertex_fn_name)));
        }
        let fragment_function_descriptor = MTL4LibraryFunctionDescriptor::new();
        {
            fragment_function_descriptor.setLibrary(Some(library));
            fragment_function_descriptor.setName(Some(&objc2_foundation::NSString::from_str(
                fragment_fn_name,
            )));
        }

        let pipeline_descriptor = MTL4RenderPipelineDescriptor::new();
        unsafe {
            pipeline_descriptor.setVertexFunctionDescriptor(Some(&vertex_function_descriptor));
            pipeline_descriptor.setFragmentFunctionDescriptor(Some(&fragment_function_descriptor));
            let color_attachment = pipeline_descriptor
                .colorAttachments()
                .objectAtIndexedSubscript(0);
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
            .unwrap_or_else(|_| panic!("metal4: compiler could not build the {vertex_fn_name}/{fragment_fn_name} pipeline state"))
    }

    /// Like `compile_pipeline`, but for `path_sprite_vertex`/`_fragment`:
    /// premultiplied-alpha "over" blending (source factors both `One`, not
    /// `compile_pipeline`'s `SourceAlpha`/`One`) — matches
    /// `path_rasterization_fragment`'s premultiplied output
    /// (`color.rgb * color.a * alpha, alpha * color.a`) and
    /// `MetalRenderer::build_path_sprite_pipeline_state`.
    fn compile_path_sprite_pipeline(
        compiler: &ProtocolObject<dyn MTL4Compiler>,
        library: &ProtocolObject<dyn objc2_metal::MTLLibrary>,
    ) -> Retained<ProtocolObject<dyn MTLRenderPipelineState>> {
        let vertex_function_descriptor = MTL4LibraryFunctionDescriptor::new();
        {
            vertex_function_descriptor.setLibrary(Some(library));
            vertex_function_descriptor.setName(Some(&objc2_foundation::NSString::from_str(
                "path_sprite_vertex",
            )));
        }
        let fragment_function_descriptor = MTL4LibraryFunctionDescriptor::new();
        {
            fragment_function_descriptor.setLibrary(Some(library));
            fragment_function_descriptor.setName(Some(&objc2_foundation::NSString::from_str(
                "path_sprite_fragment",
            )));
        }

        let pipeline_descriptor = MTL4RenderPipelineDescriptor::new();
        unsafe {
            pipeline_descriptor.setVertexFunctionDescriptor(Some(&vertex_function_descriptor));
            pipeline_descriptor.setFragmentFunctionDescriptor(Some(&fragment_function_descriptor));
            let color_attachment = pipeline_descriptor
                .colorAttachments()
                .objectAtIndexedSubscript(0);
            color_attachment.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
            color_attachment.setBlendingState(MTL4BlendState::Enabled);
            color_attachment.setRgbBlendOperation(objc2_metal::MTLBlendOperation::Add);
            color_attachment.setAlphaBlendOperation(objc2_metal::MTLBlendOperation::Add);
            color_attachment.setSourceRGBBlendFactor(objc2_metal::MTLBlendFactor::One);
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
            .expect("metal4: compiler could not build the path_sprite pipeline state")
    }

    /// Like `compile_path_sprite_pipeline`, but for
    /// `path_rasterization_vertex`/`_fragment` — same premultiplied blend
    /// idea, except the destination *alpha* factor is also
    /// `OneMinusSourceAlpha` (not `One`), because unlike the sprite pass
    /// this one can composite multiple overlapping, still-transparent path
    /// triangles into the same intermediate pixel before it's ever sampled,
    /// so alpha itself has to accumulate correctly too — matches
    /// `MetalRenderer::build_path_rasterization_pipeline_state`. Also
    /// declares the pipeline's raster sample count so it matches the MSAA
    /// intermediate texture it renders into.
    fn compile_path_rasterization_pipeline(
        compiler: &ProtocolObject<dyn MTL4Compiler>,
        library: &ProtocolObject<dyn objc2_metal::MTLLibrary>,
    ) -> Retained<ProtocolObject<dyn MTLRenderPipelineState>> {
        let vertex_function_descriptor = MTL4LibraryFunctionDescriptor::new();
        {
            vertex_function_descriptor.setLibrary(Some(library));
            vertex_function_descriptor.setName(Some(&objc2_foundation::NSString::from_str(
                "path_rasterization_vertex",
            )));
        }
        let fragment_function_descriptor = MTL4LibraryFunctionDescriptor::new();
        {
            fragment_function_descriptor.setLibrary(Some(library));
            fragment_function_descriptor.setName(Some(&objc2_foundation::NSString::from_str(
                "path_rasterization_fragment",
            )));
        }

        let pipeline_descriptor = MTL4RenderPipelineDescriptor::new();
        unsafe {
            pipeline_descriptor.setVertexFunctionDescriptor(Some(&vertex_function_descriptor));
            pipeline_descriptor.setFragmentFunctionDescriptor(Some(&fragment_function_descriptor));
            if PATH_SAMPLE_COUNT > 1 {
                pipeline_descriptor.setRasterSampleCount(PATH_SAMPLE_COUNT as usize);
            }
            let color_attachment = pipeline_descriptor
                .colorAttachments()
                .objectAtIndexedSubscript(0);
            color_attachment.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
            color_attachment.setBlendingState(MTL4BlendState::Enabled);
            color_attachment.setRgbBlendOperation(objc2_metal::MTLBlendOperation::Add);
            color_attachment.setAlphaBlendOperation(objc2_metal::MTLBlendOperation::Add);
            color_attachment.setSourceRGBBlendFactor(objc2_metal::MTLBlendFactor::One);
            color_attachment.setSourceAlphaBlendFactor(objc2_metal::MTLBlendFactor::One);
            color_attachment
                .setDestinationRGBBlendFactor(objc2_metal::MTLBlendFactor::OneMinusSourceAlpha);
            color_attachment
                .setDestinationAlphaBlendFactor(objc2_metal::MTLBlendFactor::OneMinusSourceAlpha);
        }

        compiler
            .newRenderPipelineStateWithDescriptor_compilerTaskOptions_error(
                &pipeline_descriptor,
                None,
            )
            .expect("metal4: compiler could not build the path_rasterization pipeline state")
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
            layer.set_drawable_size(core_graphics::geometry::CGSize::new(
                size.width.0 as f64,
                size.height.0 as f64,
            ));
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

    /// The current frame's command buffer — `self.command_buffers[self.
    /// current_slot.get()]`. `current_slot` is set once, at the top of
    /// `draw()`/`render_scene_to_image()`, before any of this frame's
    /// encoding happens, so every method that needs "this frame's command
    /// buffer" (`begin_main_encoder`, `draw_paths_to_intermediate`, …) can
    /// just call this instead of taking an extra parameter.
    fn command_buffer(&self) -> &ProtocolObject<dyn MTL4CommandBuffer> {
        &self.command_buffers[self.current_slot.get()]
    }

    /// Renders every `gpui::Scene` primitive kind: `scene.shadows`,
    /// `scene.paths`, `scene.quads`, `scene.underlines`,
    /// `scene.monochrome_sprites`, `scene.polychrome_sprites`, and
    /// `scene.surfaces`. Dispatches through `draw_batches`, which walks
    /// `scene.batches()` — the same painter's-algorithm iterator
    /// `MetalRenderer` itself uses — so draw order follows the scene's real
    /// z-order across primitive *types*, not just within one type: a
    /// contiguous run of same-kind primitives (and, for sprites, same-atlas-
    /// texture primitives) still becomes a single `drawPrimitives` call, but
    /// the run ends the moment the next primitive in paint order is a
    /// different kind. One deliberate simplification remains:
    /// `draw_paths_from_intermediate` always emits one composite sprite per
    /// path, which can double-blend an overlapping region of two
    /// *same-batch* transparent paths — see its doc comment.
    pub fn draw(&mut self, scene: &Scene) {
        let layer = match &self.layer {
            Some(l) => l.clone(),
            None => {
                log::error!("metal4: draw() called on a headless renderer");
                return;
            }
        };

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

        // Triple-buffered: pick this frame's allocator/instance-buffer slot
        // and only wait for the *last* frame that used it
        // (frame_number - FRAME_SLOTS), not the previous frame — see
        // `allocators`'s and `current_slot`'s field doc comments. Once
        // `frame_number > FRAME_SLOTS` every slot has been used before, so
        // that wait always applies from then on; before that, every slot is
        // still on its first, never-before-used allocator/buffers, so there
        // is nothing to wait for.
        let frame_number = self.frame_number.get() + 1;
        self.frame_number.set(frame_number);
        let slot = ((frame_number - 1) % FRAME_SLOTS as u64) as usize;
        self.current_slot.set(slot);
        if frame_number > FRAME_SLOTS as u64 {
            let previous_use = frame_number - FRAME_SLOTS as u64;
            let signaled = {
                self.shared_event
                    .waitUntilSignaledValue_timeoutMS(previous_use, 1000)
            };
            if !signaled {
                log::error!(
                    "metal4: timed out waiting for frame {previous_use} (slot {slot}'s last use) to finish on the GPU"
                );
            }
        }

        self.allocators[slot].reset();
        self.command_buffer()
            .beginCommandBufferWithAllocator(&self.allocators[slot]);

        let texture: Retained<ProtocolObject<dyn objc2_metal::MTLTexture>> =
            unsafe { bridge_retain(drawable.texture().as_ptr() as *mut c_void) };

        let alpha = if self.opaque { 1.0 } else { 0.0 };
        let encoder = self.begin_main_encoder(&texture, Some(alpha), viewport_size_px);
        let encoder = self.draw_batches(scene, viewport_size_px, &texture, encoder);

        encoder.endEncoding();
        self.command_buffer().endCommandBuffer();

        // Present choreography verified against Apple's own sample
        // (Metal4Renderer+Encoding.m's `submitCommandBuffer:toCommandQueue:forView:`):
        // wait for the drawable before commit, commit, signal the drawable
        // after commit, then present — presentation itself is decoupled
        // from the command buffer in Metal 4.
        let mtl_drawable: Retained<ProtocolObject<dyn objc2_metal::MTLDrawable>> =
            unsafe { bridge_retain(drawable.as_ptr() as *mut c_void) };
        self.queue.waitForDrawable(&mtl_drawable);

        let mut command_buffers: [NonNull<ProtocolObject<dyn MTL4CommandBuffer>>; 1] =
            [NonNull::from(self.command_buffer())];
        unsafe {
            self.queue
                .commit_count(NonNull::from(&mut command_buffers[0]), 1)
        };

        self.queue.signalDrawable(&mtl_drawable);
        drawable.present();

        self.queue
            .signalEvent_value(ProtocolObject::from_ref(&*self.shared_event), frame_number);
    }

    /// Starts a new `MTL4RenderCommandEncoder` against `texture`, viewport
    /// already set. `clear_alpha` picks the color-attachment load action:
    /// `Some(alpha)` clears to transparent-black-or-opaque-black (used once,
    /// at the top of a frame); `None` loads the attachment's existing
    /// contents instead, which is what `draw_batches` needs every time it
    /// has to end-and-restart the encoder around a `Paths` batch (Metal
    /// can't have two render command encoders active on one command buffer
    /// at once, so compositing the path-rasterization intermediate texture
    /// — its own pass, see `draw_paths_to_intermediate` — means suspending
    /// the main pass, not drawing inside it).
    fn begin_main_encoder(
        &self,
        texture: &ProtocolObject<dyn objc2_metal::MTLTexture>,
        clear_alpha: Option<f64>,
        viewport_size: Size<DevicePixels>,
    ) -> Retained<ProtocolObject<dyn MTL4RenderCommandEncoder>> {
        let render_pass_descriptor = MTL4RenderPassDescriptor::new();
        unsafe {
            let color_attachment = render_pass_descriptor
                .colorAttachments()
                .objectAtIndexedSubscript(0);
            color_attachment.setTexture(Some(texture));
            color_attachment.setStoreAction(MTLStoreAction::Store);
            match clear_alpha {
                Some(alpha) => {
                    color_attachment.setLoadAction(MTLLoadAction::Clear);
                    color_attachment.setClearColor(MTLClearColor {
                        red: 0.0,
                        green: 0.0,
                        blue: 0.0,
                        alpha,
                    });
                }
                None => color_attachment.setLoadAction(MTLLoadAction::Load),
            }
        }

        let encoder = self
            .command_buffer()
            .renderCommandEncoderWithDescriptor(&render_pass_descriptor)
            .expect("metal4: failed to create a render command encoder");

        encoder.setViewport(objc2_metal::MTLViewport {
            originX: 0.0,
            originY: 0.0,
            width: i32::from(viewport_size.width) as f64,
            height: i32::from(viewport_size.height) as f64,
            znear: 0.0,
            zfar: 1.0,
        });

        encoder
    }

    /// Grows `buffer` (if needed) then copies the *entire* `data` slice into
    /// it, starting at element 0.
    ///
    /// Must be called **at most once per frame**, before any `draw_*` call
    /// that reads `buffer`, never per-batch. `MTLBuffer` contents written
    /// via a raw CPU pointer (`contents()`, as every `copy_nonoverlapping`
    /// in this file does) are not part of the recorded command stream the
    /// way a `setVertexBuffer:offset:atIndex:` *binding* is — the GPU reads
    /// them live, only once the whole command buffer actually executes,
    /// which happens after `commit()`, i.e. after every CPU write for the
    /// *entire frame* has already happened. If two draw calls this frame
    /// both wrote their own data into the same buffer at offset 0 (which is
    /// exactly what calling a `draw_*` method once per `scene.batches()` run
    /// of its kind would do), every draw call referencing that buffer would
    /// render using whichever call's write happened to be *last*, not its
    /// own — this was the actual cause of the corrupted glyphs/quad borders
    /// `draw_batches` first shipped with. The fix: write each kind's full
    /// per-frame array here, once, and have every batch of that kind read
    /// its own slice out of the *same* already-fully-written buffer via
    /// `baseInstance`/`vertexStart` (see `draw_quads` and friends) instead
    /// of re-writing it.
    fn write_full<T: Copy>(
        &self,
        buffer_cell: &RefCell<(Retained<ProtocolObject<dyn MTLBuffer>>, usize)>,
        data: &[T],
    ) {
        let mut buffer = buffer_cell.borrow_mut();
        if data.len() > buffer.1 {
            let new_capacity = data.len().next_power_of_two();
            let new_buffer = Self::new_buffer(
                &self.mtl4_device,
                &self.instance_heap,
                &self.residency_set,
                mem::size_of::<T>() * new_capacity,
            );
            // Safe even though `buffer.0` usually never went through
            // `new_buffer`'s own fallback `addAllocation` at all (it's
            // heap-backed, covered by `instance_heap`'s single membership)
            // — `removeAllocation` on a non-member is a documented no-op
            // (see `instance_heap`'s field doc comment). This only does
            // real work, and only needs to, for the rare buffer that *did*
            // take the fallback path.
            {
                self.residency_set.removeAllocation(buffer.0.as_ref());
                self.residency_set.commit();
            }
            *buffer = (new_buffer, new_capacity);
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                buffer.0.contents().as_ptr() as *mut T,
                data.len(),
            );
        }
    }

    /// Grows `buffer` (if needed) to hold at least `min_elements`, without
    /// writing anything — the bump-allocated counterpart to `write_full`,
    /// for buffers multiple draw calls this frame each append their own
    /// slice to (`path_vertices_buffer`, `path_sprites_buffer`) rather than
    /// each holding the whole frame's data in one write. Growing has to
    /// happen once, up front, sized for the frame's *total* — growing again
    /// mid-frame would free the old buffer while an earlier-this-frame draw
    /// call still references its GPU address (see `write_full`'s doc
    /// comment for why that address is only read after every batch has been
    /// encoded, not as each batch is encoded).
    fn ensure_capacity<T>(
        &self,
        buffer_cell: &RefCell<(Retained<ProtocolObject<dyn MTLBuffer>>, usize)>,
        min_elements: usize,
    ) {
        let mut buffer = buffer_cell.borrow_mut();
        if min_elements > buffer.1 {
            let new_capacity = min_elements.next_power_of_two();
            let new_buffer = Self::new_buffer(
                &self.mtl4_device,
                &self.instance_heap,
                &self.residency_set,
                mem::size_of::<T>() * new_capacity,
            );
            // See `write_full`'s matching comment — a no-op unless `buffer.0`
            // happened to be a fallback (non-heap) buffer.
            {
                self.residency_set.removeAllocation(buffer.0.as_ref());
                self.residency_set.commit();
            }
            *buffer = (new_buffer, new_capacity);
        }
    }

    /// Walks `scene.batches()` — the same painter's-algorithm iterator
    /// `MetalRenderer::draw_primitives_to_texture` drives — dispatching each
    /// contiguous run to the matching `draw_*` method with just that run's
    /// range, so draw order matches the scene's real paint order across
    /// primitive types while same-kind (same-atlas-texture, for sprites)
    /// runs still collapse into one `drawPrimitives` call. A `Paths` batch
    /// is the one kind that can't just draw into `encoder`: rasterizing into
    /// the shared intermediate texture is its own render pass (see
    /// `draw_paths_to_intermediate`), so this ends `encoder`, runs that
    /// pass, opens a fresh encoder over the same `texture` with `Load`
    /// (preserving everything painted so far), and composites that batch's
    /// paths into it before continuing — mirroring how
    /// `MetalRenderer::draw_primitives_to_texture` ends and recreates its
    /// own encoder around each `PrimitiveBatch::Paths`. Returns the final
    /// live encoder, still open, for the caller to end.
    ///
    /// Before the loop, writes each non-empty kind's *entire* per-frame
    /// array into its instance buffer exactly once via `write_full` (or, for
    /// the two path buffers — which each batch has to append its own slice
    /// to rather than write in one shot — pre-grows them to the frame's
    /// total via `ensure_capacity` and resets their bump cursors). See
    /// `write_full`'s doc comment for why calling a `draw_*` method's own
    /// buffer-write once per *batch* instead, the way an earlier version of
    /// this method did, corrupts every draw call of that kind.
    fn draw_batches(
        &self,
        scene: &Scene,
        viewport_size: Size<DevicePixels>,
        texture: &ProtocolObject<dyn objc2_metal::MTLTexture>,
        mut encoder: Retained<ProtocolObject<dyn MTL4RenderCommandEncoder>>,
    ) -> Retained<ProtocolObject<dyn MTL4RenderCommandEncoder>> {
        let slot = self.current_slot.get();
        if !scene.shadows.is_empty() {
            self.write_full(&self.shadows_buffer[slot], &scene.shadows);
        }
        if !scene.quads.is_empty() {
            self.write_full(&self.quads_buffer[slot], &scene.quads);
        }
        if !scene.underlines.is_empty() {
            self.write_full(&self.underlines_buffer[slot], &scene.underlines);
        }
        if !scene.monochrome_sprites.is_empty() {
            self.write_full(
                &self.monochrome_sprites_buffer[slot],
                &scene.monochrome_sprites,
            );
        }
        if !scene.polychrome_sprites.is_empty() {
            self.write_full(
                &self.polychrome_sprites_buffer[slot],
                &scene.polychrome_sprites,
            );
        }
        let total_path_vertices: usize = scene.paths.iter().map(|path| path.vertices.len()).sum();
        self.ensure_capacity::<PathRasterizationVertex>(
            &self.path_vertices_buffer[slot],
            total_path_vertices,
        );
        self.ensure_capacity::<PathSprite>(&self.path_sprites_buffer[slot], scene.paths.len());
        // An upper bound, not the exact batch count (computing that exactly
        // would mean draining `scene.batches()` once just to count, then
        // again to draw) — every `MonochromeSprites`/`PolychromeSprites`
        // batch has at least one sprite, so the number of such batches can
        // never exceed the total sprite count. `ViewportSize` is 8 bytes, so
        // even a generous overshoot here costs nothing.
        let sprite_batches_upper_bound =
            scene.monochrome_sprites.len() + scene.polychrome_sprites.len();
        self.ensure_capacity::<ViewportSize>(
            &self.atlas_size_buffer[slot],
            sprite_batches_upper_bound,
        );
        self.atlas_size_cursor.set(0);
        self.path_vertices_cursor.set(0);
        self.path_sprites_cursor.set(0);

        for batch in scene.batches() {
            match batch {
                PrimitiveBatch::Shadows(range) => {
                    self.draw_shadows(range, viewport_size, &encoder);
                }
                PrimitiveBatch::Quads(range) => {
                    self.draw_quads(range, viewport_size, &encoder);
                }
                PrimitiveBatch::Paths(range) => {
                    let paths = &scene.paths[range];

                    encoder.endEncoding();
                    let did_rasterize = self.draw_paths_to_intermediate(paths, viewport_size);
                    encoder = self.begin_main_encoder(texture, None, viewport_size);

                    if did_rasterize {
                        self.draw_paths_from_intermediate(paths, viewport_size, &encoder);
                    }
                }
                PrimitiveBatch::Underlines(range) => {
                    self.draw_underlines(range, viewport_size, &encoder);
                }
                PrimitiveBatch::MonochromeSprites { range, .. } => {
                    self.draw_monochrome_sprites(
                        &scene.monochrome_sprites,
                        range,
                        viewport_size,
                        &encoder,
                    );
                }
                PrimitiveBatch::PolychromeSprites { range, .. } => {
                    self.draw_polychrome_sprites(
                        &scene.polychrome_sprites,
                        range,
                        viewport_size,
                        &encoder,
                    );
                }
                PrimitiveBatch::Surfaces(range) => {
                    self.draw_surfaces(&scene.surfaces[range], viewport_size, &encoder);
                }
                PrimitiveBatch::SubpixelSprites { .. } => unreachable!(),
            }
        }

        encoder
    }

    /// Renders a scene to an offscreen texture and reads back the pixels —
    /// no window, `CAMetalLayer`, or drawable involved. Mirrors
    /// `MetalRenderer::render_scene_to_image`, with the same
    /// `FRAME_SLOTS`-way triple-buffered allocator/command-buffer/instance-
    /// buffer rotation `draw()` uses (see the module doc comment) instead
    /// of that method's synchronous `wait_until_completed` (there's no
    /// `MTL4CommandBuffer` equivalent of that call — completion is only
    /// observable via an `MTLSharedEvent`, so this reuses the renderer's
    /// own).
    #[cfg(any(test, feature = "test-support"))]
    pub fn render_scene_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> anyhow::Result<image::RgbaImage> {
        use anyhow::bail;

        if size.width.0 <= 0 || size.height.0 <= 0 {
            bail!("metal4: invalid size for render_scene_to_image: {:?}", size);
        }

        let texture_descriptor = metal::TextureDescriptor::new();
        texture_descriptor.set_width(size.width.0 as u64);
        texture_descriptor.set_height(size.height.0 as u64);
        texture_descriptor.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        texture_descriptor
            .set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        // `Shared`, not `Managed` (which is what MetalRenderer's Metal 3
        // path uses): Shared needs no explicit CPU/GPU synchronization on
        // either unified or discrete memory, sidestepping the blit-encoder
        // sync MetalRenderer needs — encoding that blit under Metal 4 would
        // mean a whole extra MTL4ComputeCommandEncoder pass (Metal 4 folds
        // MTLBlitCommandEncoder into it), not worth it for a headless test
        // helper that isn't on any performance-sensitive path.
        texture_descriptor.set_storage_mode(metal::MTLStorageMode::Shared);
        let legacy_target_texture = self.device.new_texture(&texture_descriptor);
        let target_texture: Retained<ProtocolObject<dyn objc2_metal::MTLTexture>> =
            unsafe { bridge_retain(legacy_target_texture.as_ptr() as *mut c_void) };

        unsafe {
            self.residency_set.addAllocation(target_texture.as_ref());
            self.residency_set.commit();
            self.residency_set.requestResidency();
        }

        // See `draw()`'s matching block for why the wait target and reset
        // are slot-relative, not "the previous frame," now that
        // `FRAME_SLOTS` allocators/command buffers/instance buffers exist.
        let frame_number = self.frame_number.get() + 1;
        self.frame_number.set(frame_number);
        let slot = ((frame_number - 1) % FRAME_SLOTS as u64) as usize;
        self.current_slot.set(slot);
        if frame_number > FRAME_SLOTS as u64 {
            let previous_use = frame_number - FRAME_SLOTS as u64;
            let signaled = unsafe {
                self.shared_event
                    .waitUntilSignaledValue_timeoutMS(previous_use, 1000)
            };
            if !signaled {
                log::error!(
                    "metal4: timed out waiting for frame {previous_use} (slot {slot}'s last use) to finish on the GPU"
                );
            }
        }

        unsafe { self.allocators[slot].reset() };
        unsafe {
            self.command_buffer()
                .beginCommandBufferWithAllocator(&self.allocators[slot])
        };

        // alpha: 1.0 unconditionally, unlike draw()'s `self.opaque`-derived
        // alpha — a headless render target has no window behind it to show
        // through, so it should always come back fully opaque regardless of
        // the renderer's transparency setting.
        let encoder = self.begin_main_encoder(&target_texture, Some(1.0), size);
        let encoder = self.draw_batches(scene, size, &target_texture, encoder);

        unsafe { encoder.endEncoding() };
        unsafe { self.command_buffer().endCommandBuffer() };

        let mut command_buffers: [NonNull<ProtocolObject<dyn MTL4CommandBuffer>>; 1] =
            [NonNull::from(self.command_buffer())];
        unsafe {
            self.queue
                .commit_count(NonNull::from(&mut command_buffers[0]), 1)
        };
        unsafe {
            self.queue
                .signalEvent_value(ProtocolObject::from_ref(&*self.shared_event), frame_number)
        };
        let signaled = unsafe {
            self.shared_event
                .waitUntilSignaledValue_timeoutMS(frame_number, 5000)
        };
        if !signaled {
            bail!("metal4: timed out waiting for the headless frame to finish on the GPU");
        }

        read_texture_to_image(&legacy_target_texture)
    }

    /// Draws `range` out of `quads_buffer`, which `draw_batches` has already
    /// written *in full* for this frame via `write_full` — this method never
    /// touches the buffer's contents, only which slice of it this call's
    /// `drawPrimitives` reads, via `baseInstance`. MSL's `[[instance_id]]`
    /// (see `quad_id` in `shaders.metal`'s `quad_vertex`) is defined to
    /// start counting at `baseInstance`, so `quads[range.start]` becomes
    /// `[[instance_id]] == range.start` in the shader without the shader
    /// itself needing to know about the offset.
    fn draw_quads(
        &self,
        range: Range<usize>,
        viewport_size: Size<DevicePixels>,
        encoder: &ProtocolObject<dyn MTL4RenderCommandEncoder>,
    ) {
        if range.is_empty() {
            return;
        }
        let slot = self.current_slot.get();
        let quads_buffer = self.quads_buffer[slot].borrow();
        let viewport_size_buffer = &self.viewport_size_buffer[slot];

        unsafe {
            viewport_size_buffer
                .contents()
                .cast::<ViewportSize>()
                .as_ptr()
                .write(ViewportSize {
                    width: i32::from(viewport_size.width),
                    height: i32::from(viewport_size.height),
                });
        }

        unsafe {
            self.argument_table
                .setAddress_atIndex(self.unit_vertices.gpuAddress(), 0);
            self.argument_table
                .setAddress_atIndex(quads_buffer.0.gpuAddress(), 1);
            self.argument_table
                .setAddress_atIndex(viewport_size_buffer.gpuAddress(), 2);
            encoder.setArgumentTable_atStages(
                &self.argument_table,
                MTLRenderStages::Vertex | MTLRenderStages::Fragment,
            );
            encoder.setRenderPipelineState(&self.quad_pipeline_state);
            encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::Triangle,
                0,
                6,
                range.len(),
                range.start,
            );
        }
    }

    /// Mirrors `draw_quads` exactly — same argument-table indices
    /// (`UnderlineInputIndex` has the identical numeric layout to
    /// `QuadInputIndex`: vertices=0, primitive-data=1, viewport=2), just a
    /// different pipeline and instance type. `underlines_buffer` is already
    /// fully written for the frame by `draw_batches`.
    fn draw_underlines(
        &self,
        range: Range<usize>,
        viewport_size: Size<DevicePixels>,
        encoder: &ProtocolObject<dyn MTL4RenderCommandEncoder>,
    ) {
        if range.is_empty() {
            return;
        }
        let slot = self.current_slot.get();
        let underlines_buffer = self.underlines_buffer[slot].borrow();
        let viewport_size_buffer = &self.viewport_size_buffer[slot];

        unsafe {
            viewport_size_buffer
                .contents()
                .cast::<ViewportSize>()
                .as_ptr()
                .write(ViewportSize {
                    width: i32::from(viewport_size.width),
                    height: i32::from(viewport_size.height),
                });
        }

        unsafe {
            self.argument_table
                .setAddress_atIndex(self.unit_vertices.gpuAddress(), 0);
            self.argument_table
                .setAddress_atIndex(underlines_buffer.0.gpuAddress(), 1);
            self.argument_table
                .setAddress_atIndex(viewport_size_buffer.gpuAddress(), 2);
            encoder.setArgumentTable_atStages(
                &self.argument_table,
                MTLRenderStages::Vertex | MTLRenderStages::Fragment,
            );
            encoder.setRenderPipelineState(&self.underline_pipeline_state);
            encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::Triangle,
                0,
                6,
                range.len(),
                range.start,
            );
        }
    }

    /// Mirrors `draw_quads`/`draw_underlines` exactly — same three
    /// argument-table indices (`ShadowInputIndex` has the same numeric
    /// layout too), just the shadow pipeline and instance type.
    /// `shadows_buffer` is already fully written for the frame by
    /// `draw_batches`. The shadow shaders do all the blur/corner-radius math
    /// themselves from the raw `Shadow` fields; nothing extra to bind here.
    fn draw_shadows(
        &self,
        range: Range<usize>,
        viewport_size: Size<DevicePixels>,
        encoder: &ProtocolObject<dyn MTL4RenderCommandEncoder>,
    ) {
        if range.is_empty() {
            return;
        }
        let slot = self.current_slot.get();
        let shadows_buffer = self.shadows_buffer[slot].borrow();
        let viewport_size_buffer = &self.viewport_size_buffer[slot];

        unsafe {
            viewport_size_buffer
                .contents()
                .cast::<ViewportSize>()
                .as_ptr()
                .write(ViewportSize {
                    width: i32::from(viewport_size.width),
                    height: i32::from(viewport_size.height),
                });
        }

        unsafe {
            self.argument_table
                .setAddress_atIndex(self.unit_vertices.gpuAddress(), 0);
            self.argument_table
                .setAddress_atIndex(shadows_buffer.0.gpuAddress(), 1);
            self.argument_table
                .setAddress_atIndex(viewport_size_buffer.gpuAddress(), 2);
            encoder.setArgumentTable_atStages(
                &self.argument_table,
                MTLRenderStages::Vertex | MTLRenderStages::Fragment,
            );
            encoder.setRenderPipelineState(&self.shadow_pipeline_state);
            encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::Triangle,
                0,
                6,
                range.len(),
                range.start,
            );
        }
    }

    /// Writes `width`/`height` into the next free slot of `atlas_size_buffer`
    /// (see its field doc comment) and returns that slot's GPU address, for
    /// `draw_monochrome_sprites`/`draw_polychrome_sprites` to bind. Advances
    /// `atlas_size_cursor`; `draw_batches` already grew the buffer to fit
    /// this frame's sprite-batch count (an upper bound, so always enough)
    /// before the batch loop started, so this never needs to grow or clamp.
    fn next_atlas_size_slot(&self, width: i32, height: i32) -> u64 {
        let cursor = self.atlas_size_cursor.get();
        self.atlas_size_cursor.set(cursor + 1);
        let buffer = self.atlas_size_buffer[self.current_slot.get()].borrow();
        unsafe {
            buffer
                .0
                .contents()
                .cast::<ViewportSize>()
                .as_ptr()
                .add(cursor)
                .write(ViewportSize { width, height });
        }
        buffer.0.gpuAddress() + (cursor * mem::size_of::<ViewportSize>()) as u64
    }

    /// First primitive that samples a texture rather than just filling flat
    /// colour — everything about the buffer/argument-table plumbing mirrors
    /// `draw_quads`, but two things are genuinely new: binding the atlas
    /// texture itself (a *texture* argument-table slot, `setTexture_atIndex`
    /// with an `MTLResourceID`, a completely separate list from the buffer
    /// slots `setAddress_atIndex` populates — see the argument-table
    /// descriptor's `setMaxTextureBindCount` in `new_internal`), and making
    /// sure that texture is actually resident, since `MetalAtlas` allocates
    /// its textures with the legacy `metal` crate, entirely outside this
    /// renderer's `residency_set` — Metal 4 has no automatic residency to
    /// fall back on if that's missed (an unregistered texture would sample
    /// as garbage or fault, not just warn).
    ///
    /// Draws `range` out of `sprites` (the full `scene.monochrome_sprites`,
    /// already fully written into `monochrome_sprites_buffer` for the frame
    /// by `draw_batches` — see `write_full`'s doc comment for why this
    /// method must not write to it itself). Reads `range.start`'s
    /// `texture_id` rather than always index 0 (unlike a `draw_*` call given
    /// a plain slice, `range` here indexes into the *whole-scene* array),
    /// and warns if a later sprite in the range disagrees — that warning
    /// should never fire in practice, since `scene.batches()` never puts two
    /// different atlas textures in the same range (see `BatchIterator::
    /// next`'s `texture_id == texture_id` guard); kept as a defensive check
    /// rather than an `assert!`/unwrap, since a mis-batched call here would
    /// otherwise silently sample the wrong tile.
    fn draw_monochrome_sprites(
        &self,
        sprites: &[MonochromeSprite],
        range: Range<usize>,
        viewport_size: Size<DevicePixels>,
        encoder: &ProtocolObject<dyn MTL4RenderCommandEncoder>,
    ) {
        if range.is_empty() {
            return;
        }
        let texture_id = sprites[range.start].tile.texture_id;
        if sprites[range.clone()]
            .iter()
            .any(|sprite| sprite.tile.texture_id != texture_id)
        {
            log::warn!(
                "metal4: monochrome sprites span more than one atlas texture; only {:?}'s sprites will sample the right texture",
                texture_id
            );
        }

        let Some(legacy_texture) = self.sprite_atlas.metal_texture(texture_id) else {
            log::warn!("metal4: skipping sprites whose atlas texture {texture_id:?} was released");
            return;
        };
        let texture: Retained<ProtocolObject<dyn MTLTexture>> =
            unsafe { bridge_retain(legacy_texture.as_ptr() as *mut c_void) };

        if self.resident_atlas_texture.get() != Some(texture_id) {
            {
                self.residency_set.addAllocation(texture.as_ref());
                self.residency_set.commit();
                self.residency_set.requestResidency();
            }
            self.resident_atlas_texture.set(Some(texture_id));
        }

        let slot = self.current_slot.get();
        let sprites_buffer = self.monochrome_sprites_buffer[slot].borrow();
        let viewport_size_buffer = &self.viewport_size_buffer[slot];

        unsafe {
            viewport_size_buffer
                .contents()
                .cast::<ViewportSize>()
                .as_ptr()
                .write(ViewportSize {
                    width: i32::from(viewport_size.width),
                    height: i32::from(viewport_size.height),
                });
        }
        let atlas_size_address =
            self.next_atlas_size_slot(texture.width() as i32, texture.height() as i32);

        unsafe {
            self.argument_table
                .setAddress_atIndex(self.unit_vertices.gpuAddress(), 0);
            self.argument_table
                .setAddress_atIndex(sprites_buffer.0.gpuAddress(), 1);
            self.argument_table
                .setAddress_atIndex(viewport_size_buffer.gpuAddress(), 2);
            self.argument_table
                .setAddress_atIndex(atlas_size_address, 3);
            self.argument_table
                .setTexture_atIndex(texture.gpuResourceID(), 4);
            encoder.setArgumentTable_atStages(
                &self.argument_table,
                MTLRenderStages::Vertex | MTLRenderStages::Fragment,
            );
            encoder.setRenderPipelineState(&self.monochrome_sprite_pipeline_state);
            encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::Triangle,
                0,
                6,
                range.len(),
                range.start,
            );
        }
    }

    /// Same texture-binding/residency mechanics as `draw_monochrome_sprites`
    /// (see its doc comment), just `PolychromeSprite`/the polychrome
    /// pipeline and a BGRA8Unorm atlas texture instead of monochrome's
    /// single-channel A8Unorm one. The fragment shader samples the atlas
    /// directly as the sprite's final colour (modulated by `opacity` and a
    /// corner-radius SDF) rather than using it as an alpha mask over a
    /// separate tint colour — there's no `MonochromeSprite`-style `color`
    /// field here at all.
    fn draw_polychrome_sprites(
        &self,
        sprites: &[PolychromeSprite],
        range: Range<usize>,
        viewport_size: Size<DevicePixels>,
        encoder: &ProtocolObject<dyn MTL4RenderCommandEncoder>,
    ) {
        if range.is_empty() {
            return;
        }
        let texture_id = sprites[range.start].tile.texture_id;
        if sprites[range.clone()]
            .iter()
            .any(|sprite| sprite.tile.texture_id != texture_id)
        {
            log::warn!(
                "metal4: polychrome sprites span more than one atlas texture; only {:?}'s sprites will sample the right texture",
                texture_id
            );
        }

        let Some(legacy_texture) = self.sprite_atlas.metal_texture(texture_id) else {
            log::warn!("metal4: skipping sprites whose atlas texture {texture_id:?} was released");
            return;
        };
        let texture: Retained<ProtocolObject<dyn MTLTexture>> =
            unsafe { bridge_retain(legacy_texture.as_ptr() as *mut c_void) };

        if self.resident_atlas_texture.get() != Some(texture_id) {
            {
                self.residency_set.addAllocation(texture.as_ref());
                self.residency_set.commit();
                self.residency_set.requestResidency();
            }
            self.resident_atlas_texture.set(Some(texture_id));
        }

        let slot = self.current_slot.get();
        let sprites_buffer = self.polychrome_sprites_buffer[slot].borrow();
        let viewport_size_buffer = &self.viewport_size_buffer[slot];

        unsafe {
            viewport_size_buffer
                .contents()
                .cast::<ViewportSize>()
                .as_ptr()
                .write(ViewportSize {
                    width: i32::from(viewport_size.width),
                    height: i32::from(viewport_size.height),
                });
        }
        let atlas_size_address =
            self.next_atlas_size_slot(texture.width() as i32, texture.height() as i32);

        unsafe {
            self.argument_table
                .setAddress_atIndex(self.unit_vertices.gpuAddress(), 0);
            self.argument_table
                .setAddress_atIndex(sprites_buffer.0.gpuAddress(), 1);
            self.argument_table
                .setAddress_atIndex(viewport_size_buffer.gpuAddress(), 2);
            self.argument_table
                .setAddress_atIndex(atlas_size_address, 3);
            self.argument_table
                .setTexture_atIndex(texture.gpuResourceID(), 4);
            encoder.setArgumentTable_atStages(
                &self.argument_table,
                MTLRenderStages::Vertex | MTLRenderStages::Fragment,
            );
            encoder.setRenderPipelineState(&self.polychrome_sprite_pipeline_state);
            encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::Triangle,
                0,
                6,
                range.len(),
                range.start,
            );
        }
    }

    /// Rebuilds `path_intermediate_texture`/`path_intermediate_msaa_texture`
    /// for a new viewport size — mirrors
    /// `MetalRenderer::update_path_intermediate_textures` exactly (down to
    /// the memoryless-on-Apple-GPUs storage-mode choice for the MSAA
    /// texture, since MSAA render targets are resolved within a single pass
    /// and never need to persist past it), just called lazily from
    /// `draw_paths_to_intermediate` — only when there's a path to
    /// rasterize — rather than eagerly on every resize.
    fn rebuild_path_intermediate_textures(&self, size: Size<DevicePixels>) {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            *self.path_intermediate_texture.borrow_mut() = None;
            *self.path_intermediate_msaa_texture.borrow_mut() = None;
            self.path_intermediate_size.set(None);
            return;
        }

        let texture_descriptor = metal::TextureDescriptor::new();
        texture_descriptor.set_width(size.width.0 as u64);
        texture_descriptor.set_height(size.height.0 as u64);
        texture_descriptor.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        texture_descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        texture_descriptor
            .set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        let new_intermediate = self.device.new_texture(&texture_descriptor);

        if PATH_SAMPLE_COUNT > 1 {
            let is_apple_gpu = self.device.supports_family(metal::MTLGPUFamily::Apple1);
            let storage_mode = if is_apple_gpu {
                metal::MTLStorageMode::Memoryless
            } else {
                metal::MTLStorageMode::Private
            };
            let msaa_descriptor = texture_descriptor;
            msaa_descriptor.set_texture_type(metal::MTLTextureType::D2Multisample);
            msaa_descriptor.set_storage_mode(storage_mode);
            msaa_descriptor.set_sample_count(PATH_SAMPLE_COUNT as _);
            *self.path_intermediate_msaa_texture.borrow_mut() =
                Some(self.device.new_texture(&msaa_descriptor));
        } else {
            *self.path_intermediate_msaa_texture.borrow_mut() = None;
        }

        *self.path_intermediate_texture.borrow_mut() = Some(new_intermediate);
        self.path_intermediate_size
            .set(Some((size.width.0, size.height.0)));
    }

    /// Rasterizes one `Paths` batch (a contiguous, same-order run from
    /// `scene.batches()`, not necessarily every path in the scene) into the
    /// shared, full-viewport-sized intermediate texture — its own render
    /// pass, entirely separate from the main one, since Metal (3 or 4 alike)
    /// can't switch render targets mid-encoder. `draw_batches` ends the main
    /// encoder before calling this and opens a fresh one afterwards (see its
    /// doc comment), so the composite this batch's `draw_paths_from_
    /// intermediate` call performs lands at the right point in paint order
    /// relative to whatever was drawn before and after it, mirroring how
    /// `MetalRenderer::draw_primitives_to_texture` ends/recreates its own
    /// encoder around each `PrimitiveBatch::Paths`.
    ///
    /// Returns whether anything was actually rasterized — mirrors
    /// `MetalRenderer::draw_paths_to_intermediate`'s `did_draw` bool, so the
    /// caller knows whether it's safe to skip the composite pass entirely.
    ///
    /// Writes this batch's flattened vertices into `path_vertices_buffer`
    /// starting at `path_vertices_cursor`, not offset 0 — a second `Paths`
    /// batch in the same frame must not overwrite the first batch's
    /// vertices before the GPU has actually read them (see `write_full`'s
    /// doc comment); `draw_batches` pre-grows the buffer to the whole
    /// frame's vertex total up front so this never needs to reallocate.
    fn draw_paths_to_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        viewport_size: Size<DevicePixels>,
    ) -> bool {
        if paths.is_empty() {
            return false;
        }

        if self.path_intermediate_size.get()
            != Some((viewport_size.width.0, viewport_size.height.0))
        {
            self.rebuild_path_intermediate_textures(viewport_size);
        }

        let intermediate_texture_ref = self.path_intermediate_texture.borrow();
        let Some(intermediate_texture) = intermediate_texture_ref.as_ref() else {
            return false;
        };

        let mut vertices = Vec::new();
        for path in paths {
            vertices.extend(path.vertices.iter().map(|v| PathRasterizationVertex {
                xy_position: v.xy_position,
                st_position: v.st_position,
                color: path.color,
                bounds: path.bounds.intersect(&path.content_mask.bounds),
            }));
        }
        if vertices.is_empty() {
            return false;
        }

        // `draw_batches` already grew `path_vertices_buffer` to fit the
        // whole frame's total vertex count via `ensure_capacity`, so this
        // only ever appends at the current cursor — never grows (which
        // would free the buffer while an earlier `Paths` batch this same
        // frame still has an already-encoded rasterization draw pointing at
        // its old GPU address; see `write_full`'s doc comment) and never
        // overwrites an earlier batch's still-unread vertices at offset 0
        // the way a single shared buffer written from scratch each call
        // would.
        let slot = self.current_slot.get();
        let base_vertex = self.path_vertices_cursor.get();
        self.path_vertices_cursor.set(base_vertex + vertices.len());
        let path_vertices_buffer = self.path_vertices_buffer[slot].borrow();
        unsafe {
            std::ptr::copy_nonoverlapping(
                vertices.as_ptr(),
                (path_vertices_buffer.0.contents().as_ptr() as *mut PathRasterizationVertex)
                    .add(base_vertex),
                vertices.len(),
            );
        }

        let viewport_size_buffer = &self.viewport_size_buffer[slot];
        unsafe {
            viewport_size_buffer
                .contents()
                .cast::<ViewportSize>()
                .as_ptr()
                .write(ViewportSize {
                    width: i32::from(viewport_size.width),
                    height: i32::from(viewport_size.height),
                });
        }

        let msaa_texture_ref = self.path_intermediate_msaa_texture.borrow();
        let render_pass_descriptor = MTL4RenderPassDescriptor::new();
        unsafe {
            let color_attachment = render_pass_descriptor
                .colorAttachments()
                .objectAtIndexedSubscript(0);
            color_attachment.setLoadAction(MTLLoadAction::Clear);
            color_attachment.setClearColor(MTLClearColor {
                red: 0.0,
                green: 0.0,
                blue: 0.0,
                alpha: 0.0,
            });

            let resolve_texture: Retained<ProtocolObject<dyn MTLTexture>> =
                bridge_retain(intermediate_texture.as_ptr() as *mut c_void);
            if let Some(msaa_texture) = msaa_texture_ref.as_ref() {
                let msaa_bridged: Retained<ProtocolObject<dyn MTLTexture>> =
                    bridge_retain(msaa_texture.as_ptr() as *mut c_void);
                color_attachment.setTexture(Some(&msaa_bridged));
                color_attachment.setResolveTexture(Some(&resolve_texture));
                color_attachment.setStoreAction(MTLStoreAction::MultisampleResolve);
            } else {
                color_attachment.setTexture(Some(&resolve_texture));
                color_attachment.setStoreAction(MTLStoreAction::Store);
            }
        }

        let encoder = self
            .command_buffer()
            .renderCommandEncoderWithDescriptor(&render_pass_descriptor)
            .expect("metal4: failed to create a render command encoder for path rasterization");

        unsafe {
            self.argument_table
                .setAddress_atIndex(path_vertices_buffer.0.gpuAddress(), 0);
            self.argument_table
                .setAddress_atIndex(viewport_size_buffer.gpuAddress(), 1);
            encoder.setArgumentTable_atStages(
                &self.argument_table,
                MTLRenderStages::Vertex | MTLRenderStages::Fragment,
            );
            encoder.setRenderPipelineState(&self.path_rasterization_pipeline_state);
            encoder.drawPrimitives_vertexStart_vertexCount_instanceCount(
                MTLPrimitiveType::Triangle,
                base_vertex,
                vertices.len(),
                1,
            );
            encoder.endEncoding();
        }

        true
    }

    /// Composites this batch's slice of the shared intermediate texture —
    /// just rasterized by `draw_paths_to_intermediate`, in the render pass
    /// `draw_batches` ran immediately before reopening the main encoder —
    /// into that main render target, one `PathSprite` quad per path.
    ///
    /// Simplification not present in `MetalRenderer`: always emits one
    /// sprite per path, rather than reasoning about draw order to sometimes
    /// merge several into a single spanning-rect copy (see
    /// `MetalRenderer::draw_paths_from_intermediate`'s comment on why it
    /// does that — "each pixel must only be copied once, in case of
    /// transparent paths"). Two *different-order* paths in the same batch
    /// whose bounds overlap would get composited, and blended against the
    /// destination, once each here, which can double-blend the overlap
    /// region — a real, known gap, acceptable for now because it only
    /// matters for overlapping transparent paths that also happen to fall
    /// in the same contiguous paint-order run.
    fn draw_paths_from_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        viewport_size: Size<DevicePixels>,
        encoder: &ProtocolObject<dyn MTL4RenderCommandEncoder>,
    ) {
        if paths.is_empty() {
            return;
        }
        let intermediate_texture_ref = self.path_intermediate_texture.borrow();
        let Some(intermediate_texture) = intermediate_texture_ref.as_ref() else {
            log::error!("metal4: no path intermediate texture to composite from");
            return;
        };
        let texture: Retained<ProtocolObject<dyn MTLTexture>> =
            unsafe { bridge_retain(intermediate_texture.as_ptr() as *mut c_void) };

        // Idempotent even though this can now run more than once per frame
        // (one `Paths` batch's worth each time): residency-set membership is
        // additive, so re-adding an already-resident texture is harmless.
        {
            self.residency_set.addAllocation(texture.as_ref());
            self.residency_set.commit();
            self.residency_set.requestResidency();
        }

        let sprites: Vec<PathSprite> = paths
            .iter()
            .map(|path| PathSprite {
                bounds: path.clipped_bounds(),
            })
            .collect();

        // `draw_batches` already grew `path_sprites_buffer` to fit
        // `scene.paths.len()` via `ensure_capacity`, so — same reasoning as
        // `draw_paths_to_intermediate`'s vertex write — this only ever
        // appends at the current cursor, never overwriting an earlier
        // `Paths` batch's still-unread sprites this frame.
        let slot = self.current_slot.get();
        let base_sprite = self.path_sprites_cursor.get();
        self.path_sprites_cursor.set(base_sprite + sprites.len());
        let path_sprites_buffer = self.path_sprites_buffer[slot].borrow();
        unsafe {
            std::ptr::copy_nonoverlapping(
                sprites.as_ptr(),
                (path_sprites_buffer.0.contents().as_ptr() as *mut PathSprite).add(base_sprite),
                sprites.len(),
            );
        }

        let viewport_size_buffer = &self.viewport_size_buffer[slot];
        unsafe {
            viewport_size_buffer
                .contents()
                .cast::<ViewportSize>()
                .as_ptr()
                .write(ViewportSize {
                    width: i32::from(viewport_size.width),
                    height: i32::from(viewport_size.height),
                });
        }

        unsafe {
            self.argument_table
                .setAddress_atIndex(self.unit_vertices.gpuAddress(), 0);
            self.argument_table
                .setAddress_atIndex(path_sprites_buffer.0.gpuAddress(), 1);
            self.argument_table
                .setAddress_atIndex(viewport_size_buffer.gpuAddress(), 2);
            self.argument_table
                .setTexture_atIndex(texture.gpuResourceID(), 4);
            encoder.setArgumentTable_atStages(
                &self.argument_table,
                MTLRenderStages::Vertex | MTLRenderStages::Fragment,
            );
            encoder.setRenderPipelineState(&self.path_sprite_pipeline_state);
            encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::Triangle,
                0,
                6,
                sprites.len(),
                base_sprite,
            );
        }
    }

    /// Video surfaces are macOS-only (`PaintSurface` carries no pixel buffer on iOS).
    #[cfg(target_os = "ios")]
    fn draw_surfaces(
        &self,
        _surfaces: &[PaintSurface],
        _viewport_size: Size<DevicePixels>,
        _encoder: &ProtocolObject<dyn MTL4RenderCommandEncoder>,
    ) {
    }

    /// One `PaintSurface` (a video frame) at a time, unlike every other
    /// primitive — each has its own pair of Y/CbCr textures pulled fresh
    /// from `core_video_texture_cache` every call, so there's no batching
    /// win to instancing them together the way quads/sprites are (mirrors
    /// `MetalRenderer::draw_surfaces`' own per-surface loop).
    ///
    /// Two real Metal-4-specific concerns beyond the mechanical
    /// translation: the Y/CbCr textures are argument-table-bound
    /// (`setTexture_atIndex`) like atlas/path-intermediate textures, so
    /// they need residency too — and because they're brand new objects
    /// every call (a fresh video frame each time, never "the same texture
    /// as last call" the way `draw_monochrome_sprites`/
    /// `draw_polychrome_sprites` can assume for their atlas textures), this
    /// removes last call's two textures from `residency_set` before adding
    /// this call's, rather than letting stale entries accumulate forever.
    #[cfg(target_os = "macos")]
    fn draw_surfaces(
        &self,
        surfaces: &[PaintSurface],
        viewport_size: Size<DevicePixels>,
        encoder: &ProtocolObject<dyn MTL4RenderCommandEncoder>,
    ) {
        if surfaces.is_empty() {
            return;
        }

        for texture in self.surface_textures_in_residency.borrow_mut().drain(..) {
            self.residency_set.removeAllocation(texture.as_ref());
        }

        let slot = self.current_slot.get();
        unsafe {
            self.viewport_size_buffer[slot]
                .contents()
                .cast::<ViewportSize>()
                .as_ptr()
                .write(ViewportSize {
                    width: i32::from(viewport_size.width),
                    height: i32::from(viewport_size.height),
                });
        }

        let mut newly_resident = Vec::with_capacity(surfaces.len() * 2);

        for surface in surfaces {
            if surface.image_buffer.get_pixel_format()
                != kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
            {
                log::error!(
                    "metal4: surface pixel buffer is not 420YpCbCr8BiPlanarFullRange, skipping"
                );
                continue;
            }

            let texture_size = gpui::size(
                DevicePixels::from(surface.image_buffer.get_width() as i32),
                DevicePixels::from(surface.image_buffer.get_height() as i32),
            );

            let Ok(y_texture_cv) = self.core_video_texture_cache.create_texture_from_image(
                surface.image_buffer.as_concrete_TypeRef(),
                None,
                metal::MTLPixelFormat::R8Unorm,
                surface.image_buffer.get_width_of_plane(0),
                surface.image_buffer.get_height_of_plane(0),
                0,
            ) else {
                log::error!("metal4: failed to create a Metal texture for the surface's Y plane");
                continue;
            };
            let Ok(cb_cr_texture_cv) = self.core_video_texture_cache.create_texture_from_image(
                surface.image_buffer.as_concrete_TypeRef(),
                None,
                metal::MTLPixelFormat::RG8Unorm,
                surface.image_buffer.get_width_of_plane(1),
                surface.image_buffer.get_height_of_plane(1),
                1,
            ) else {
                log::error!(
                    "metal4: failed to create a Metal texture for the surface's CbCr plane"
                );
                continue;
            };

            // CVMetalTextureGetTexture is a "get" accessor, not a create/
            // copy — the returned object is owned by y_texture_cv/
            // cb_cr_texture_cv (kept alive for this loop iteration), so
            // bridge_retain's real retain (not the autoreleased-return-value
            // fast path bridge_retain_autoreleased assumes, which this
            // plain C FFI call doesn't participate in) is the correct,
            // if slightly conservative, way to hold our own reference.
            let y_texture: Retained<ProtocolObject<dyn MTLTexture>> = unsafe {
                bridge_retain(
                    CVMetalTextureGetTexture(y_texture_cv.as_concrete_TypeRef()) as *mut c_void
                )
            };
            let cb_cr_texture: Retained<ProtocolObject<dyn MTLTexture>> = unsafe {
                bridge_retain(
                    CVMetalTextureGetTexture(cb_cr_texture_cv.as_concrete_TypeRef()) as *mut c_void,
                )
            };

            {
                self.residency_set.addAllocation(y_texture.as_ref());
                self.residency_set.addAllocation(cb_cr_texture.as_ref());
                self.residency_set.commit();
                self.residency_set.requestResidency();
            }
            newly_resident.push(y_texture.clone());
            newly_resident.push(cb_cr_texture.clone());

            let bounds = SurfaceBounds {
                bounds: surface.bounds,
                content_mask: surface.content_mask,
            };

            unsafe {
                std::ptr::copy_nonoverlapping(
                    &bounds as *const SurfaceBounds,
                    self.surfaces_buffer.borrow().contents().as_ptr() as *mut SurfaceBounds,
                    1,
                );
                self.atlas_size_buffer[slot]
                    .borrow()
                    .0
                    .contents()
                    .cast::<ViewportSize>()
                    .as_ptr()
                    .write(ViewportSize {
                        width: i32::from(texture_size.width),
                        height: i32::from(texture_size.height),
                    });

                self.argument_table
                    .setAddress_atIndex(self.unit_vertices.gpuAddress(), 0);
                self.argument_table
                    .setAddress_atIndex(self.surfaces_buffer.borrow().gpuAddress(), 1);
                self.argument_table
                    .setAddress_atIndex(self.viewport_size_buffer[slot].gpuAddress(), 2);
                self.argument_table
                    .setAddress_atIndex(self.atlas_size_buffer[slot].borrow().0.gpuAddress(), 3);
                self.argument_table
                    .setTexture_atIndex(y_texture.gpuResourceID(), 4);
                self.argument_table
                    .setTexture_atIndex(cb_cr_texture.gpuResourceID(), 5);
                encoder.setArgumentTable_atStages(
                    &self.argument_table,
                    MTLRenderStages::Vertex | MTLRenderStages::Fragment,
                );
                encoder.setRenderPipelineState(&self.surface_pipeline_state);
                encoder.drawPrimitives_vertexStart_vertexCount_instanceCount(
                    MTLPrimitiveType::Triangle,
                    0,
                    6,
                    1,
                );
            }
        }

        *self.surface_textures_in_residency.borrow_mut() = newly_resident;
    }
}

/// Duplicated from `metal_renderer.rs`'s private helper of the same name
/// rather than widening that file's visibility — this keeps `draw()`'s file
/// (a real upstream-churn hotspot, see the module doc comment) at zero
/// touches from this renderer.
#[cfg(any(test, feature = "test-support"))]
fn read_texture_to_image(texture: &metal::TextureRef) -> anyhow::Result<image::RgbaImage> {
    use anyhow::Context as _;

    let width = texture.width() as u32;
    let height = texture.height() as u32;
    let bytes_per_row = width as usize * 4;
    let mut pixels = vec![0u8; height as usize * bytes_per_row];

    let region = metal::MTLRegion {
        origin: metal::MTLOrigin { x: 0, y: 0, z: 0 },
        size: metal::MTLSize {
            width: width as u64,
            height: height as u64,
            depth: 1,
        },
    };
    texture.get_bytes(
        pixels.as_mut_ptr() as *mut std::ffi::c_void,
        bytes_per_row as u64,
        region,
        0,
    );

    // Convert BGRA to RGBA (swap B and R channels)
    for chunk in pixels.chunks_exact_mut(4) {
        chunk.swap(0, 2);
    }

    image::RgbaImage::from_raw(width, height, pixels)
        .context("failed to create RgbaImage from pixel data")
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{Bounds, Corners, PlatformAtlas, Point, ScaledPixels, hsla, point, size};

    /// Renders three solid-colour quads (red / green / blue, at known
    /// positions with a gap between them) through the real Metal 4 path —
    /// no window, no `CAMetalLayer`, no app skin/vibrancy — and writes the
    /// result to a PNG for visual inspection, in addition to sampling
    /// specific pixels to assert the colours actually landed where
    /// expected. This is deliberately isolated from the full app (whose
    /// glass/vibrancy chrome makes it hard to tell what the renderer itself
    /// produced vs. what the window material drew on top).
    #[test]
    fn quads_render_at_expected_positions_and_colors() {
        let canvas = size(600i32, 400i32);
        let canvas_scaled = size(
            ScaledPixels(canvas.width as f32),
            ScaledPixels(canvas.height as f32),
        );
        let full_mask = gpui::ContentMask {
            bounds: Bounds::new(Point::default(), canvas_scaled),
        };

        let make_quad = |x: f32, y: f32, w: f32, h: f32, color: gpui::Hsla| Quad {
            bounds: Bounds::new(
                point(ScaledPixels(x), ScaledPixels(y)),
                size(ScaledPixels(w), ScaledPixels(h)),
            ),
            content_mask: full_mask,
            background: color.into(),
            corner_radii: Corners::default(),
            ..Default::default()
        };

        let red = hsla(0.0, 1.0, 0.5, 1.0);
        let green = hsla(0.33, 1.0, 0.5, 1.0);
        let blue = hsla(0.66, 1.0, 0.5, 1.0);

        let mut scene = Scene::default();
        scene.quads.push(make_quad(20.0, 20.0, 150.0, 150.0, red));
        scene
            .quads
            .push(make_quad(225.0, 20.0, 150.0, 150.0, green));
        scene.quads.push(make_quad(430.0, 20.0, 150.0, 150.0, blue));

        let mut renderer = Metal4Renderer::new_headless();
        let image = renderer
            .render_scene_to_image(&scene, size(canvas.width.into(), canvas.height.into()))
            .expect("metal4 headless render failed");

        let out_path = std::env::temp_dir().join("metal4_quads_test.png");
        image
            .save(&out_path)
            .expect("failed to save metal4 quads test PNG");
        eprintln!("metal4 quads test image written to {}", out_path.display());

        assert_eq!(image.width(), canvas.width as u32);
        assert_eq!(image.height(), canvas.height as u32);

        // Centers of each quad.
        assert_pixel_matches(&image, 95, 95, red, "red quad");
        assert_pixel_matches(&image, 300, 95, green, "green quad");
        assert_pixel_matches(&image, 505, 95, blue, "blue quad");

        // The gaps between quads and the area below them should be the
        // clear color (opaque black, since `new_headless` passes `true`
        // for `opaque` — see `new_internal`).
        let black = hsla(0.0, 0.0, 0.0, 1.0);
        assert_pixel_matches(&image, 190, 95, black, "gap between red and green");
        assert_pixel_matches(&image, 300, 300, black, "below the quads");
    }

    /// Renders a single straight (non-wavy) underline through the real
    /// Metal 4 path, mirroring `quads_render_at_expected_positions_and_colors`
    /// — this is `draw_underlines`'s first exercise, added right after the
    /// quad milestone since `UnderlineInputIndex` has the exact same
    /// numeric layout as `QuadInputIndex` and the pipeline/buffer plumbing
    /// is a near-verbatim copy of the quad path (see its doc comment).
    #[test]
    fn underlines_render_at_expected_positions_and_colors() {
        let canvas = size(400i32, 100i32);
        let canvas_scaled = size(
            ScaledPixels(canvas.width as f32),
            ScaledPixels(canvas.height as f32),
        );
        let full_mask = gpui::ContentMask {
            bounds: Bounds::new(Point::default(), canvas_scaled),
        };

        let yellow = hsla(0.16, 1.0, 0.5, 1.0);

        let underline = Underline {
            order: 0,
            pad: 0,
            bounds: Bounds::new(
                point(ScaledPixels(50.0), ScaledPixels(40.0)),
                size(ScaledPixels(300.0), ScaledPixels(4.0)),
            ),
            content_mask: full_mask,
            color: yellow,
            thickness: ScaledPixels(4.0),
            wavy: false.into(),
        };

        let mut scene = Scene::default();
        scene.underlines.push(underline);

        let mut renderer = Metal4Renderer::new_headless();
        let image = renderer
            .render_scene_to_image(&scene, size(canvas.width.into(), canvas.height.into()))
            .expect("metal4 headless render failed");

        let out_path = std::env::temp_dir().join("metal4_underline_test.png");
        image
            .save(&out_path)
            .expect("failed to save metal4 underline test PNG");
        eprintln!(
            "metal4 underline test image written to {}",
            out_path.display()
        );

        assert_pixel_matches(&image, 200, 41, yellow, "underline, mid-span");
        assert_pixel_matches(&image, 60, 41, yellow, "underline, near left end");

        let black = hsla(0.0, 0.0, 0.0, 1.0);
        assert_pixel_matches(&image, 200, 10, black, "above the underline");
        assert_pixel_matches(&image, 200, 80, black, "below the underline");
    }

    /// Renders a single hard-edged (`blur_radius: 0`), unrounded shadow
    /// through the real Metal 4 path — same pattern as the quad/underline
    /// tests. `blur_radius: 0` takes shadow_fragment's `quad_sdf` branch
    /// instead of its Gaussian-blur one, giving a crisp rectangle that's
    /// easy to pixel-test precisely; the blur math itself is exercised by
    /// every real shadow the app draws once this ships, just not asserted
    /// pixel-by-pixel here.
    #[test]
    fn shadows_render_at_expected_positions_and_colors() {
        let canvas = size(300i32, 300i32);
        let canvas_scaled = size(
            ScaledPixels(canvas.width as f32),
            ScaledPixels(canvas.height as f32),
        );
        let full_mask = gpui::ContentMask {
            bounds: Bounds::new(Point::default(), canvas_scaled),
        };

        let purple = hsla(0.75, 1.0, 0.5, 1.0);
        let shadow_bounds = Bounds::new(
            point(ScaledPixels(75.0), ScaledPixels(75.0)),
            size(ScaledPixels(150.0), ScaledPixels(150.0)),
        );

        let shadow = Shadow {
            order: 0,
            blur_radius: ScaledPixels(0.0),
            bounds: shadow_bounds,
            corner_radii: Corners::default(),
            content_mask: full_mask,
            color: purple,
            element_bounds: shadow_bounds,
            element_corner_radii: Corners::default(),
            inset: 0,
            pad: 0,
        };

        let mut scene = Scene::default();
        scene.shadows.push(shadow);

        let mut renderer = Metal4Renderer::new_headless();
        let image = renderer
            .render_scene_to_image(&scene, size(canvas.width.into(), canvas.height.into()))
            .expect("metal4 headless render failed");

        let out_path = std::env::temp_dir().join("metal4_shadow_test.png");
        image
            .save(&out_path)
            .expect("failed to save metal4 shadow test PNG");
        eprintln!("metal4 shadow test image written to {}", out_path.display());

        assert_pixel_matches(&image, 150, 150, purple, "shadow center");
        assert_pixel_matches(&image, 80, 80, purple, "shadow, near top-left corner");
        assert_pixel_matches(&image, 219, 219, purple, "shadow, near bottom-right corner");

        let black = hsla(0.0, 0.0, 0.0, 1.0);
        assert_pixel_matches(&image, 20, 20, black, "outside the shadow, top-left");
        assert_pixel_matches(&image, 280, 280, black, "outside the shadow, bottom-right");
    }

    /// Renders a single monochrome sprite through the real Metal 4 path —
    /// the first primitive that samples a texture rather than just filling
    /// flat colour. Uploads a real tile into `Metal4Renderer`'s own
    /// `MetalAtlas` (via the `PlatformAtlas` trait, the same interface
    /// gpui's text/icon system uses) rather than reaching around it, so
    /// this exercises the actual atlas-texture residency and
    /// `MTL4ArgumentTable` texture-binding path, not a shortcut.
    ///
    /// The uploaded tile is a solid, fully-opaque 32x32 square (alpha=255
    /// everywhere) — `AtlasTextureKind::Monochrome` tiles are single-channel
    /// `A8Unorm`, and `monochrome_sprite_fragment` multiplies the sprite's
    /// colour by the sampled alpha, so this should render as a plain
    /// `sprite.color`-tinted rectangle, same shape as the quad test's
    /// squares but by a completely different GPU path.
    #[test]
    fn monochrome_sprites_render_at_expected_positions_and_colors() {
        let canvas = size(200i32, 200i32);
        let canvas_scaled = size(
            ScaledPixels(canvas.width as f32),
            ScaledPixels(canvas.height as f32),
        );
        let full_mask = gpui::ContentMask {
            bounds: Bounds::new(Point::default(), canvas_scaled),
        };

        let mut renderer = Metal4Renderer::new_headless();

        let tile_size = gpui::size(DevicePixels(32), DevicePixels(32));
        let key = gpui::AtlasKey::Glyph(gpui::RenderGlyphParams {
            font_id: gpui::FontId(0),
            glyph_id: gpui::GlyphId(0),
            font_size: gpui::px(32.0),
            subpixel_variant: Point::default(),
            scale_factor: 1.0,
            is_emoji: false,
            subpixel_rendering: false,
            dilation: 0,
        });
        let opaque_pixels = vec![255u8; (tile_size.width.0 * tile_size.height.0) as usize];
        let tile = renderer
            .sprite_atlas()
            .get_or_insert_with(key, &mut || {
                Ok(Some((
                    tile_size,
                    std::borrow::Cow::Borrowed(opaque_pixels.as_slice()),
                )))
            })
            .expect("atlas upload failed")
            .expect("atlas upload returned no tile");
        assert_eq!(
            tile.texture_id.kind,
            gpui::AtlasTextureKind::Monochrome,
            "a non-emoji, non-subpixel glyph key should land in the monochrome atlas"
        );

        let cyan = hsla(0.5, 1.0, 0.5, 1.0);

        let sprite = MonochromeSprite {
            order: 0,
            pad: 0,
            bounds: Bounds::new(
                point(ScaledPixels(60.0), ScaledPixels(60.0)),
                size(ScaledPixels(80.0), ScaledPixels(80.0)),
            ),
            content_mask: full_mask,
            color: cyan,
            tile,
            transformation: gpui::TransformationMatrix::unit(),
        };

        let mut scene = Scene::default();
        scene.monochrome_sprites.push(sprite);

        let image = renderer
            .render_scene_to_image(&scene, size(canvas.width.into(), canvas.height.into()))
            .expect("metal4 headless render failed");

        let out_path = std::env::temp_dir().join("metal4_monochrome_sprite_test.png");
        image
            .save(&out_path)
            .expect("failed to save metal4 monochrome sprite test PNG");
        eprintln!(
            "metal4 monochrome sprite test image written to {}",
            out_path.display()
        );

        assert_pixel_matches(&image, 100, 100, cyan, "sprite center");
        assert_pixel_matches(&image, 65, 65, cyan, "sprite, near top-left corner");
        assert_pixel_matches(&image, 134, 134, cyan, "sprite, near bottom-right corner");

        let black = hsla(0.0, 0.0, 0.0, 1.0);
        assert_pixel_matches(&image, 10, 10, black, "outside the sprite, top-left");
        assert_pixel_matches(&image, 190, 190, black, "outside the sprite, bottom-right");
    }

    /// Renders a single polychrome sprite (an image, not a glyph/icon)
    /// through the real Metal 4 path — same texture-binding/residency
    /// mechanics as the monochrome sprite test, but the atlas tile carries
    /// the sprite's actual colour (BGRA8Unorm) rather than being an alpha
    /// mask over a separately-specified tint. `AtlasKey::Image` is what
    /// routes a tile into the polychrome atlas (`AtlasKey::texture_kind()`);
    /// unlike the monochrome test's fake `RenderGlyphParams`, an `ImageId`
    /// needs no fake font/glyph fields to construct.
    #[test]
    fn polychrome_sprites_render_at_expected_positions_and_colors() {
        let canvas = size(200i32, 200i32);
        let canvas_scaled = size(
            ScaledPixels(canvas.width as f32),
            ScaledPixels(canvas.height as f32),
        );
        let full_mask = gpui::ContentMask {
            bounds: Bounds::new(Point::default(), canvas_scaled),
        };

        let mut renderer = Metal4Renderer::new_headless();

        let tile_size = gpui::size(DevicePixels(32), DevicePixels(32));
        let key = gpui::AtlasKey::Image(gpui::RenderImageParams {
            image_id: gpui::ImageId(0),
            frame_index: 0,
        });
        // BGRA8Unorm, opaque solid orange (matches AtlasTextureKind::
        // Polychrome's pixel format — see metal_atlas.rs).
        let orange_bgra: [u8; 4] = [0, 140, 255, 255];
        let pixels: Vec<u8> = orange_bgra
            .iter()
            .copied()
            .cycle()
            .take((tile_size.width.0 * tile_size.height.0 * 4) as usize)
            .collect();
        let tile = renderer
            .sprite_atlas()
            .get_or_insert_with(key, &mut || {
                Ok(Some((
                    tile_size,
                    std::borrow::Cow::Borrowed(pixels.as_slice()),
                )))
            })
            .expect("atlas upload failed")
            .expect("atlas upload returned no tile");
        assert_eq!(
            tile.texture_id.kind,
            gpui::AtlasTextureKind::Polychrome,
            "an AtlasKey::Image should land in the polychrome atlas"
        );

        let orange = hsla(0.09, 1.0, 0.5, 1.0);

        let sprite = PolychromeSprite {
            order: 0,
            pad: 0,
            grayscale: false.into(),
            opacity: 1.0,
            bounds: Bounds::new(
                point(ScaledPixels(60.0), ScaledPixels(60.0)),
                size(ScaledPixels(80.0), ScaledPixels(80.0)),
            ),
            content_mask: full_mask,
            corner_radii: Corners::default(),
            tile,
        };

        let mut scene = Scene::default();
        scene.polychrome_sprites.push(sprite);

        let image = renderer
            .render_scene_to_image(&scene, size(canvas.width.into(), canvas.height.into()))
            .expect("metal4 headless render failed");

        let out_path = std::env::temp_dir().join("metal4_polychrome_sprite_test.png");
        image
            .save(&out_path)
            .expect("failed to save metal4 polychrome sprite test PNG");
        eprintln!(
            "metal4 polychrome sprite test image written to {}",
            out_path.display()
        );

        assert_pixel_matches(&image, 100, 100, orange, "sprite center");
        assert_pixel_matches(&image, 65, 65, orange, "sprite, near top-left corner");
        assert_pixel_matches(&image, 134, 134, orange, "sprite, near bottom-right corner");

        let black = hsla(0.0, 0.0, 0.0, 1.0);
        assert_pixel_matches(&image, 10, 10, black, "outside the sprite, top-left");
        assert_pixel_matches(&image, 190, 190, black, "outside the sprite, bottom-right");
    }

    /// Renders a single solid-filled triangle path through the real Metal 4
    /// path — the two-pass rasterize-then-composite architecture, the one
    /// genuinely different shape of problem among the primitives this
    /// renderer handles (everything else is a single pass). Built with
    /// `gpui::Path`'s own public `line_to` API (the same one gpui's own
    /// drawing code uses), not by hand-constructing vertex data, so this
    /// also exercises the real straight-line triangulation path (each
    /// straight edge contributes `st=(0,1)` uniformly, which
    /// `path_rasterization_fragment`'s near-zero-derivative branch turns
    /// into flat, fully-opaque fill with no curve antialiasing math — a
    /// deliberately simple case to pixel-test precisely, same reasoning as
    /// the shadow test's `blur_radius: 0`).
    #[test]
    fn paths_render_at_expected_positions_and_colors() {
        let canvas = size(200i32, 200i32);
        let canvas_scaled = size(
            ScaledPixels(canvas.width as f32),
            ScaledPixels(canvas.height as f32),
        );

        let magenta = hsla(0.83, 1.0, 0.5, 1.0);

        let mut path = gpui::Path::new(gpui::point(gpui::px(40.0), gpui::px(40.0)));
        path.line_to(gpui::point(gpui::px(160.0), gpui::px(40.0)));
        path.line_to(gpui::point(gpui::px(100.0), gpui::px(160.0)));
        path.color = magenta.into();
        path.content_mask = gpui::ContentMask {
            bounds: Bounds::new(
                Point::default(),
                size(
                    gpui::px(canvas.width as f32),
                    gpui::px(canvas.height as f32),
                ),
            ),
        };
        let path = path.scale(1.0);
        assert_eq!(
            path.content_mask.bounds.size, canvas_scaled,
            "sanity check: scale(1.0) should leave the full-canvas content mask numerically unchanged"
        );

        let mut scene = Scene::default();
        scene.paths.push(path);

        let mut renderer = Metal4Renderer::new_headless();
        let image = renderer
            .render_scene_to_image(&scene, size(canvas.width.into(), canvas.height.into()))
            .expect("metal4 headless render failed");

        let out_path = std::env::temp_dir().join("metal4_path_test.png");
        image
            .save(&out_path)
            .expect("failed to save metal4 path test PNG");
        eprintln!("metal4 path test image written to {}", out_path.display());

        // Well inside the triangle (40,40)-(160,40)-(100,160): near its centroid.
        assert_pixel_matches(&image, 100, 100, magenta, "triangle interior");

        let black = hsla(0.0, 0.0, 0.0, 1.0);
        assert_pixel_matches(
            &image,
            10,
            10,
            black,
            "outside the triangle, top-left corner",
        );
        assert_pixel_matches(
            &image,
            190,
            190,
            black,
            "outside the triangle, bottom-right corner",
        );
        assert_pixel_matches(&image, 100, 190, black, "below the triangle's apex");
    }

    /// Renders a single video-frame surface through the real Metal 4 path —
    /// the last of the six primitives, and the only one backed by a real
    /// `CVPixelBuffer` (biplanar 4:2:0 YCbCr, full range) rather than
    /// anything gpui's `Scene` types construct directly. Fills the whole
    /// buffer with `Y=0xFF` (full luma) and neutral chroma
    /// (`Cb=Cr=0x80`) — an achromatic value that reduces to plain grayscale
    /// under *any* correctly-designed YCbCr matrix, avoiding the need to
    /// hand-derive `surface_fragment`'s conversion matrix precisely just to
    /// pick a test colour — so the rendered surface should come out white
    /// (or very close: `Cb=Cr=128/255=0.50196`, not exactly the
    /// mathematically neutral `0.5`, a negligible few-thousandths error).
    #[test]
    fn surfaces_render_at_expected_positions_and_colors() {
        let canvas = size(200i32, 200i32);
        let canvas_scaled = size(
            ScaledPixels(canvas.width as f32),
            ScaledPixels(canvas.height as f32),
        );
        let full_mask = gpui::ContentMask {
            bounds: Bounds::new(Point::default(), canvas_scaled),
        };

        let frame_width = 64usize;
        let frame_height = 64usize;
        // Without kCVPixelBufferMetalCompatibilityKey, the buffer's backing
        // IOSurface isn't Metal-shareable and CVMetalTextureCache's
        // create_texture_from_image fails outright (CVReturn -6660) — a
        // real bug this test caught on its first run, not a hypothetical.
        let options = core_foundation::dictionary::CFDictionary::from_CFType_pairs(&[(
            core_foundation::string::CFString::from(
                core_video::pixel_buffer::CVPixelBufferKeys::MetalCompatibility,
            ),
            core_foundation::boolean::CFBoolean::true_value().as_CFType(),
        )]);
        let pixel_buffer = core_video::pixel_buffer::CVPixelBuffer::new(
            kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
            frame_width,
            frame_height,
            Some(&options),
        )
        .expect("failed to create test CVPixelBuffer");

        pixel_buffer.lock_base_address(0);
        unsafe {
            let y_base = pixel_buffer.get_base_address_of_plane(0) as *mut u8;
            let y_bytes_per_row = pixel_buffer.get_bytes_per_row_of_plane(0);
            let y_width = pixel_buffer.get_width_of_plane(0);
            let y_height = pixel_buffer.get_height_of_plane(0);
            for row in 0..y_height {
                std::ptr::write_bytes(y_base.add(row * y_bytes_per_row), 0xFF, y_width);
            }

            let cb_cr_base = pixel_buffer.get_base_address_of_plane(1) as *mut u8;
            let cb_cr_bytes_per_row = pixel_buffer.get_bytes_per_row_of_plane(1);
            // Width of plane 1, in CbCr *pairs* — the plane itself is 2
            // bytes (one Cb, one Cr) per pair, biplanar 4:2:0.
            let cb_cr_pair_count = pixel_buffer.get_width_of_plane(1);
            let cb_cr_height = pixel_buffer.get_height_of_plane(1);
            for row in 0..cb_cr_height {
                std::ptr::write_bytes(
                    cb_cr_base.add(row * cb_cr_bytes_per_row),
                    0x80,
                    cb_cr_pair_count * 2,
                );
            }
        }
        pixel_buffer.unlock_base_address(0);

        let surface = PaintSurface {
            order: 0,
            bounds: Bounds::new(
                point(ScaledPixels(60.0), ScaledPixels(60.0)),
                size(ScaledPixels(80.0), ScaledPixels(80.0)),
            ),
            content_mask: full_mask,
            image_buffer: pixel_buffer,
        };

        let mut scene = Scene::default();
        scene.surfaces.push(surface);

        let mut renderer = Metal4Renderer::new_headless();
        let image = renderer
            .render_scene_to_image(&scene, size(canvas.width.into(), canvas.height.into()))
            .expect("metal4 headless render failed");

        let out_path = std::env::temp_dir().join("metal4_surface_test.png");
        image
            .save(&out_path)
            .expect("failed to save metal4 surface test PNG");
        eprintln!(
            "metal4 surface test image written to {}",
            out_path.display()
        );

        let white = hsla(0.0, 0.0, 1.0, 1.0);
        assert_pixel_matches(&image, 100, 100, white, "surface center");
        assert_pixel_matches(&image, 65, 65, white, "surface, near top-left corner");
        assert_pixel_matches(&image, 134, 134, white, "surface, near bottom-right corner");

        let black = hsla(0.0, 0.0, 0.0, 1.0);
        assert_pixel_matches(&image, 10, 10, black, "outside the surface, top-left");
        assert_pixel_matches(&image, 190, 190, black, "outside the surface, bottom-right");
    }
}

#[cfg(test)]
fn assert_pixel_matches(
    image: &image::RgbaImage,
    x: u32,
    y: u32,
    expected: gpui::Hsla,
    label: &str,
) {
    let expected_rgba = expected.to_rgb();
    let pixel = image.get_pixel(x, y);
    let diff = |a: u8, b: f32| (a as f32 - b * 255.0).abs();
    assert!(
        diff(pixel[0], expected_rgba.r) < 10.0
            && diff(pixel[1], expected_rgba.g) < 10.0
            && diff(pixel[2], expected_rgba.b) < 10.0,
        "{label} at ({x},{y}): got {:?}, expected ~{:?}",
        pixel,
        expected_rgba
    );
}

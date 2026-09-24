use crate::{
    AbsoluteLength, App, Bounds, DefiniteLength, Edges, GridTemplate, Length, Pixels, Point, Size,
    Style, Window, size,
    util::{
        ceil_to_device_pixel, round_half_toward_zero, round_stroke_to_device_pixel,
        round_to_device_pixel,
    },
};
use collections::{FxHashMap, FxHashSet};
use std::{fmt::Debug, ops::Range};
use taffy::{
    TaffyTree, TraversePartialTree as _,
    geometry::{Point as TaffyPoint, Rect as TaffyRect, Size as TaffySize},
    prelude::{max_content, min_content},
    style::AvailableSpace as TaffyAvailableSpace,
    tree::NodeId,
};

#[cfg(feature = "stacker")]
type StackSafe<T> = stacksafe::StackSafe<T>;
#[cfg(not(feature = "stacker"))]
type StackSafe<T> = T;

type MeasureFn =
    dyn FnMut(Size<Option<Pixels>>, Size<AvailableSpace>, &mut Window, &mut App) -> Size<Pixels>;
type NodeMeasureFn = StackSafe<Box<MeasureFn>>;

struct NodeContext {
    measure: NodeMeasureFn,
    /// The most recent (known dimensions, available space) taffy asked this leaf to measure
    /// with, in device pixels. Replayed into the *current* frame's measure closure when the
    /// node is reused without taffy re-measuring it (see `TaffyLayoutEngine::pending_remeasure`).
    last_args: Option<(TaffySize<Option<f32>>, TaffySize<TaffyAvailableSpace>)>,
}
/// Perf instrumentation: (frames, nodes, layout ns, measure calls, measure ns).
static STATS: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// Returns and resets the layout counters since the last call.
static STATS_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Turns the layout counters on (off by default: they add per-call overhead).
pub fn enable_layout_stats(on: bool) {
    STATS_ENABLED.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn take_layout_stats() -> [u64; 5] {
    std::array::from_fn(|i| STATS[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// Perf instrumentation: (duration ns, subtree node count) per compute_layout call.
pub static CALLS: std::sync::Mutex<Vec<(u64, u32)>> = std::sync::Mutex::new(Vec::new());

/// Drains the per-call records.
pub fn take_layout_calls() -> Vec<(u64, u32)> {
    std::mem::take(&mut *CALLS.lock().unwrap())
}

pub struct TaffyLayoutEngine {
    taffy: TaffyTree<NodeContext>,
    absolute_layout_bounds: FxHashMap<LayoutId, Bounds<Pixels>>,
    /// Unrounded absolute border-box top-left per-node coordinate in device pixels.
    absolute_outer_origins: FxHashMap<LayoutId, Point<f32>>,
    computed_layouts: FxHashSet<LayoutId>,
    layout_bounds_scratch_space: Vec<LayoutId>,

    // Retained layout. gpui rebuilds its element tree every frame, which used to mean rebuilding
    // the taffy tree from scratch too, so taffy's per-node layout cache never survived a frame
    // and every frame was a full relayout. Instead, nodes are hash-consed across frames: a node
    // requested with the same style, the same (themselves reused) children and, for measured
    // leaves, the same content key as a node from the previous frame *is* that node, cache and
    // all. Only nodes whose inputs changed (and their ancestors) are recomputed.
    retain: bool,
    /// Previous frame's nodes by structural key, available for reuse this frame.
    pool: FxHashMap<u64, Vec<NodeId>>,
    /// Nodes handed out this frame (created or reused) with their key, `0` = not poolable.
    live: Vec<(u64, NodeId)>,
    /// Reused measured leaves that taffy may not re-measure (a layout cache hit skips the
    /// callback). Their fresh closure is replayed with `last_args` so per-frame element state
    /// (e.g. shaped text) is populated for paint.
    pending_remeasure: Vec<NodeId>,
    /// `DTB_KE_LAYOUT_VERIFY=1`: after every layout, recompute from scratch and compare.
    verify: bool,
    /// (created, reused) node counts since the last `take_reuse_stats`.
    reuse_counts: (u64, u64),
}

/// Perf instrumentation: (nodes created, nodes reused) since the last call.
pub static REUSE: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Returns and resets taffy's (cache hits, cache misses) counters.
pub fn take_taffy_cache_stats() -> (u64, u64) {
    taffy::compute::take_cache_stats()
}

/// Returns and resets the (created, reused) node counters.
pub fn take_reuse_stats() -> (u64, u64) {
    (
        REUSE[0].swap(0, std::sync::atomic::Ordering::Relaxed),
        REUSE[1].swap(0, std::sync::atomic::Ordering::Relaxed),
    )
}

fn env_flag(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(v) => !(v == "0" || v.eq_ignore_ascii_case("false")),
        Err(_) => default,
    }
}

/// A cheap fingerprint of the layout-relevant parts of a style. Collisions are harmless — a
/// candidate node is always verified with a full `==` — so this only needs to spread well.
fn style_fingerprint(style: &taffy::Style) -> u64 {
    use std::hash::{Hash, Hasher};
    use std::mem::discriminant;
    let mut h = collections::FxHasher::default();
    let len = |c: taffy::style::CompactLength, h: &mut collections::FxHasher| {
        c.tag().hash(h);
        c.value().to_bits().hash(h);
    };
    discriminant(&style.display).hash(&mut h);
    discriminant(&style.position).hash(&mut h);
    discriminant(&style.flex_direction).hash(&mut h);
    discriminant(&style.flex_wrap).hash(&mut h);
    discriminant(&style.align_items).hash(&mut h);
    discriminant(&style.align_self).hash(&mut h);
    discriminant(&style.justify_content).hash(&mut h);
    discriminant(&style.align_content).hash(&mut h);
    discriminant(&style.overflow.x).hash(&mut h);
    discriminant(&style.overflow.y).hash(&mut h);
    style.flex_grow.to_bits().hash(&mut h);
    style.flex_shrink.to_bits().hash(&mut h);
    len(style.flex_basis.into_raw(), &mut h);
    for d in [
        style.size.width,
        style.size.height,
        style.min_size.width,
        style.min_size.height,
        style.max_size.width,
        style.max_size.height,
    ] {
        len(d.into_raw(), &mut h);
    }
    for l in [
        style.margin.left,
        style.margin.right,
        style.margin.top,
        style.margin.bottom,
        style.inset.left,
        style.inset.right,
        style.inset.top,
        style.inset.bottom,
    ] {
        len(l.into_raw(), &mut h);
    }
    for l in [
        style.padding.left,
        style.padding.right,
        style.padding.top,
        style.padding.bottom,
        style.border.left,
        style.border.right,
        style.border.top,
        style.border.bottom,
        style.gap.width,
        style.gap.height,
    ] {
        len(l.into_raw(), &mut h);
    }
    h.finish()
}

const EXPECT_MESSAGE: &str = "we should avoid taffy layout errors by construction if possible";

impl TaffyLayoutEngine {
    pub fn new() -> Self {
        let mut taffy = TaffyTree::new();
        taffy.disable_rounding();
        TaffyLayoutEngine {
            taffy,
            absolute_layout_bounds: FxHashMap::default(),
            absolute_outer_origins: FxHashMap::default(),
            computed_layouts: FxHashSet::default(),
            layout_bounds_scratch_space: Vec::new(),
            retain: env_flag("DTB_KE_LAYOUT_RETAIN", true),
            pool: FxHashMap::default(),
            live: Vec::new(),
            pending_remeasure: Vec::new(),
            verify: env_flag("DTB_KE_LAYOUT_VERIFY", false),
            reuse_counts: (0, 0),
        }
    }

    pub fn clear(&mut self) {
        if self.retain {
            // Sweep: whatever last frame's pool still holds was not reused this frame.
            for (_, nodes) in self.pool.drain() {
                for node in nodes {
                    self.taffy.remove_detached(node);
                }
            }
            for (key, node) in self.live.drain(..) {
                if key != 0 {
                    self.pool.entry(key).or_default().push(node);
                }
            }
            self.pending_remeasure.clear();
        } else {
            self.taffy.clear();
        }
        self.absolute_layout_bounds.clear();
        self.absolute_outer_origins.clear();
        self.computed_layouts.clear();
    }

    /// Takes a reusable node out of last frame's pool, if one has exactly this key.
    fn take_reusable(
        &mut self,
        key: u64,
        style: &taffy::Style,
        children: &[LayoutId],
        measured: bool,
    ) -> Option<NodeId> {
        let bucket = self.pool.get_mut(&key)?;
        let position = bucket.iter().position(|&candidate| {
            self.taffy.get_node_context(candidate).is_some() == measured
                && self.taffy.style(candidate).is_ok_and(|s| s == style)
                && self.taffy.child_ids(candidate).eq(LayoutId::to_taffy_slice(children).iter().copied())
        })?;
        Some(bucket.swap_remove(position))
    }

    fn structural_key(style: &taffy::Style, children: &[LayoutId], content_key: u64) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = collections::FxHasher::default();
        style_fingerprint(style).hash(&mut h);
        content_key.hash(&mut h);
        for child in children {
            child.hash(&mut h);
        }
        // `0` is reserved for "not poolable".
        h.finish() | 1
    }

    pub fn request_layout(
        &mut self,
        style: Style,
        rem_size: Pixels,
        scale_factor: f32,
        children: &[LayoutId],
    ) -> LayoutId {
        let taffy_style = style.to_taffy(rem_size, scale_factor);

        let mut key = 0;
        if self.retain {
            key = Self::structural_key(&taffy_style, children, 0);
            if let Some(node) = self.take_reusable(key, &taffy_style, children, false) {
                self.live.push((key, node));
                self.reuse_counts.1 += 1;
                return node.into();
            }
        }

        let node = if children.is_empty() {
            self.taffy.new_leaf(taffy_style).expect(EXPECT_MESSAGE)
        } else {
            // A reused child still lists its previous (now dead) parent; re-parenting is done by
            // `new_with_children`, and `remove_detached` never clobbers the new link.
            self.taffy
                // This is safe because LayoutId is repr(transparent) to taffy::tree::NodeId.
                .new_with_children(taffy_style, LayoutId::to_taffy_slice(children))
                .expect(EXPECT_MESSAGE)
        };
        self.live.push((key, node));
        self.reuse_counts.0 += 1;
        node.into()
    }

    pub fn request_measured_layout(
        &mut self,
        style: Style,
        rem_size: Pixels,
        scale_factor: f32,
        measure: impl FnMut(
            Size<Option<Pixels>>,
            Size<AvailableSpace>,
            &mut Window,
            &mut App,
        ) -> Size<Pixels>
        + 'static,
    ) -> LayoutId {
        self.request_measured_layout_keyed(style, rem_size, scale_factor, None, measure)
    }

    /// Like [`Self::request_measured_layout`], but with a `content_key` that must change whenever
    /// anything the measure closure depends on (other than the available space, which taffy's own
    /// cache already accounts for) changes. Nodes with a key are reused across frames.
    pub fn request_measured_layout_keyed(
        &mut self,
        style: Style,
        rem_size: Pixels,
        scale_factor: f32,
        content_key: Option<u64>,
        measure: impl FnMut(
            Size<Option<Pixels>>,
            Size<AvailableSpace>,
            &mut Window,
            &mut App,
        ) -> Size<Pixels>
        + 'static,
    ) -> LayoutId {
        let taffy_style = style.to_taffy(rem_size, scale_factor);
        let measure = Box::new(measure) as Box<MeasureFn>;
        #[cfg(feature = "stacker")]
        let measure = StackSafe::new(measure);

        let mut key = 0;
        if self.retain
            && let Some(content_key) = content_key
        {
            key = Self::structural_key(&taffy_style, &[], content_key);
            if let Some(node) = self.take_reusable(key, &taffy_style, &[], true) {
                // Keep the node (and its layout cache) but swap in this frame's closure, which
                // owns this frame's element state.
                self.taffy
                    .get_node_context_mut(node)
                    .expect("reused measured node has a context")
                    .measure = measure;
                self.pending_remeasure.push(node);
                self.live.push((key, node));
                self.reuse_counts.1 += 1;
                return node.into();
            }
        }

        let node = self
            .taffy
            .new_leaf_with_context(
                taffy_style,
                NodeContext {
                    measure,
                    last_args: None,
                },
            )
            .expect(EXPECT_MESSAGE);
        self.live.push((key, node));
        self.reuse_counts.0 += 1;
        node.into()
    }

    /// Treats any `auto` dimension of the given node's style as filling `size`.
    ///
    /// This is applied to window roots before layout so they behave like the
    /// root element on the web, which stretches to fill the initial containing
    /// block (the viewport) unless given an explicit size. Explicitly styled
    /// dimensions are preserved.
    pub fn stretch_auto_size_to_fill(
        &mut self,
        id: LayoutId,
        size: Size<Pixels>,
        scale_factor: f32,
    ) {
        let style = self.taffy.style(id.0).expect(EXPECT_MESSAGE);
        let stretch_width = style.size.width.is_auto();
        let stretch_height = style.size.height.is_auto();
        if !stretch_width && !stretch_height {
            return;
        }
        let mut style = style.clone();
        if stretch_width {
            style.size.width =
                taffy::style::Dimension::length(round_to_device_pixel(size.width.0, scale_factor));
        }
        if stretch_height {
            style.size.height =
                taffy::style::Dimension::length(round_to_device_pixel(size.height.0, scale_factor));
        }
        self.taffy.set_style(id.0, style).expect(EXPECT_MESSAGE);
    }

    // Used to understand performance
    #[allow(dead_code)]
    fn count_all_children(&self, parent: LayoutId) -> anyhow::Result<u32> {
        let mut count = 0;

        for child in self.taffy.children(parent.0)? {
            // Count this child.
            count += 1;

            // Count all of this child's children.
            count += self.count_all_children(LayoutId(child))?
        }

        Ok(count)
    }

    // Used to understand performance
    #[allow(dead_code)]
    fn max_depth(&self, depth: u32, parent: LayoutId) -> anyhow::Result<u32> {
        println!(
            "{parent:?} at depth {depth} has {} children",
            self.taffy.child_count(parent.0)
        );

        let mut max_child_depth = 0;

        for child in self.taffy.children(parent.0)? {
            max_child_depth = std::cmp::max(max_child_depth, self.max_depth(0, LayoutId(child))?);
        }

        Ok(depth + 1 + max_child_depth)
    }

    // Used to understand performance
    #[allow(dead_code)]
    fn get_edges(&self, parent: LayoutId) -> anyhow::Result<Vec<(LayoutId, LayoutId)>> {
        let mut edges = Vec::new();

        for child in self.taffy.children(parent.0)? {
            edges.push((parent, LayoutId(child)));

            edges.extend(self.get_edges(LayoutId(child))?);
        }

        Ok(edges)
    }

    #[cfg_attr(feature = "stacker", stacksafe::stacksafe)]
    pub fn compute_layout(
        &mut self,
        id: LayoutId,
        available_space: Size<AvailableSpace>,
        window: &mut Window,
        cx: &mut App,
    ) {
        // Leaving this here until we have a better instrumentation approach.
        // println!("Laying out {} children", self.count_all_children(id)?);
        // println!("Max layout depth: {}", self.max_depth(0, id)?);

        // Output the edges (branches) of the tree in Mermaid format for visualization.
        // println!("Edges:");
        // for (a, b) in self.get_edges(id)? {
        //     println!("N{} --> N{}", u64::from(a), u64::from(b));
        // }
        //

        if !self.computed_layouts.insert(id) {
            let stack = &mut self.layout_bounds_scratch_space;
            stack.push(id);
            while let Some(id) = stack.pop() {
                self.absolute_layout_bounds.remove(&id);
                self.absolute_outer_origins.remove(&id);
                stack.extend(
                    self.taffy
                        .children(id.into())
                        .expect(EXPECT_MESSAGE)
                        .into_iter()
                        .map(LayoutId::from),
                );
            }
        }

        let scale_factor = window.scale_factor();

        let transform = |v: AvailableSpace| match v {
            AvailableSpace::Definite(pixels) => {
                AvailableSpace::Definite(Pixels(pixels.0 * scale_factor))
            }
            AvailableSpace::MinContent => AvailableSpace::MinContent,
            AvailableSpace::MaxContent => AvailableSpace::MaxContent,
        };
        let available_space = size(
            transform(available_space.width),
            transform(available_space.height),
        );

        use std::sync::atomic::Ordering::Relaxed;
        let stats = STATS_ENABLED.load(Relaxed);
        if stats {
            STATS[0].fetch_add(1, Relaxed);
            STATS[1].fetch_add(self.taffy.total_node_count() as u64, Relaxed);

            REUSE[0].fetch_add(std::mem::take(&mut self.reuse_counts.0), Relaxed);
            REUSE[1].fetch_add(std::mem::take(&mut self.reuse_counts.1), Relaxed);
        }

        // Reused measured leaves: taffy will skip the measure callback on a layout-cache hit, but
        // paint needs this frame's element state (shaped text) populated, so replay each one's
        // last known measure inputs through the fresh closure.
        for node in std::mem::take(&mut self.pending_remeasure) {
            if let Some(context) = self.taffy.get_node_context_mut(node)
                && let Some((known, available)) = context.last_args
            {
                measure_node(context, known, available, scale_factor, window, cx);
            }
        }

        let started = std::time::Instant::now();
        self.run_layout(id, available_space, scale_factor, stats, window, cx);

        if self.verify {
            self.verify_against_scratch(id, available_space, scale_factor, window, cx);
        }
        if stats {
            let ns = started.elapsed().as_nanos() as u64;
            STATS[2].fetch_add(ns, Relaxed);
            let subtree = self.count_all_children(id).unwrap_or(0) + 1;
            CALLS.lock().unwrap().push((ns, subtree));
        }
    }

    fn run_layout(
        &mut self,
        id: LayoutId,
        available_space: Size<AvailableSpace>,
        scale_factor: f32,
        stats: bool,
        window: &mut Window,
        cx: &mut App,
    ) {
        use std::sync::atomic::Ordering::Relaxed;
        self.taffy
            .compute_layout_with_measure(
                id.into(),
                available_space.into(),
                |known_dimensions, available_space, _id, node_context, _style| {
                    let Some(node_context) = node_context else {
                        return taffy::geometry::Size::default();
                    };
                    node_context.last_args = Some((known_dimensions, available_space));
                    let m_start = stats.then(std::time::Instant::now);
                    let measured = measure_node(
                        node_context,
                        known_dimensions,
                        available_space,
                        scale_factor,
                        window,
                        cx,
                    );
                    if let Some(m_start) = m_start {
                        STATS[3].fetch_add(1, Relaxed);
                        STATS[4].fetch_add(m_start.elapsed().as_nanos() as u64, Relaxed);
                    }
                    measured
                },
            )
            .expect(EXPECT_MESSAGE);
    }

    /// `DTB_KE_LAYOUT_VERIFY=1`: throw away every cached result under `id`, recompute from
    /// scratch, and log any node whose layout differs from the incrementally computed one.
    fn verify_against_scratch(
        &mut self,
        id: LayoutId,
        available_space: Size<AvailableSpace>,
        scale_factor: f32,
        window: &mut Window,
        cx: &mut App,
    ) {
        let mut nodes = Vec::new();
        let mut stack = vec![NodeId::from(id)];
        while let Some(node) = stack.pop() {
            nodes.push(node);
            stack.extend(self.taffy.child_ids(node));
        }
        let before: Vec<taffy::Layout> = nodes
            .iter()
            .map(|n| *self.taffy.layout(*n).expect(EXPECT_MESSAGE))
            .collect();
        for node in &nodes {
            self.taffy.mark_dirty(*node).expect(EXPECT_MESSAGE);
        }
        self.run_layout(id, available_space, scale_factor, false, window, cx);
        let mut bad = 0;
        for (node, old) in nodes.iter().zip(&before) {
            let new = self.taffy.layout(*node).expect(EXPECT_MESSAGE);
            let close = |a: f32, b: f32| (a - b).abs() < 0.01;
            if !(close(old.size.width, new.size.width)
                && close(old.size.height, new.size.height)
                && close(old.location.x, new.location.x)
                && close(old.location.y, new.location.y))
            {
                if bad < 5 {
                    log::error!(
                        "layout verify: node {node:?} differs — retained {:?}/{:?}, scratch {:?}/{:?}",
                        old.location,
                        old.size,
                        new.location,
                        new.size
                    );
                }
                bad += 1;
            }
        }
        if bad > 0 {
            log::error!("layout verify: {bad}/{} nodes differ from a scratch layout", nodes.len());
        } else {
            log::debug!("layout verify: {} nodes OK", nodes.len());
        }
    }

    // Pixel snapping
    //
    // Painting primitives at non-integer pixel coordinates produces blurry
    // output. Pixel snapping converts layout coordinates into integer
    // device-pixel coordinates so painted edges land exactly on physical
    // pixel boundaries.
    //
    // Non-integer coordinates can arise for several reasons, including:
    //   - flex distribution, percentages, centering, and text measurement
    //     can produce fractional element sizes and positions;
    //   - at fractional scale factors (for example 125% or 150%), integer
    //     logical-pixel values can map to non-integer device-pixel values.
    //
    // We pixel-snap by rounding in device-pixel space, after multiplying
    // by `scale_factor`, so that snapping targets physical pixels. Bounds
    // are divided by `scale_factor` before being returned to GPUI.
    //
    // Midpoints are rounded toward zero. This is a stylistic choice: a
    // 1-logical-pixel line at 150% scale should render as 1 dp rather than
    // 2 dp.
    //
    // Pixel snapping is done in two phases:
    //
    //  1. Pre-layout metric snapping. Before Taffy computes layout, all
    //     authored absolute lengths are rounded in `to_taffy`. This
    //     includes borders, padding, gaps, and explicit sizes.
    //     Custom-measured leaf nodes have their measured sizes rounded up
    //     to integer device-pixel lengths.
    //
    //  2. Post-layout edge snapping. After Taffy resolves the tree, layout
    //     relationships such as flex shares, grid tracks, percentages, and
    //     centering can produce new fractional edge positions. Boxes now
    //     have edges in absolute coordinates, and snapping must decide
    //     where those edges land on the device-pixel grid.
    //
    // Ideally, post-layout snapping would satisfy:
    //
    //  - Edge closure. Two raw layout edges at the same absolute position
    //    should snap to the same pixel column.
    //  - Translation stability. A component's internal geometry should not
    //    change when it moves to a new absolute position.
    //
    // These goals are in tension because rounding is not associative.
    // The simple local schemes make different tradeoffs:
    //
    //  - Absolute edge rounding gives each window coordinate one answer,
    //    so coincident edges always close globally. But a span's snapped
    //    length is `round(far) - round(near)`, which may change by 1 dp
    //    as its absolute origin moves.
    //
    //  - Parent-relative edge rounding rounds each child inside its
    //    parent's coordinate space. This guarantees translation stability,
    //    but a shared edge reached through different parents can
    //    accumulate different rounding, causing non-closure between
    //    cousins.
    //
    //  - Length rounding rounds each width, height, and thickness
    //    independently and then places boxes from those rounded lengths.
    //    Sizes stay stable under translation, but neighboring boxes derive
    //    their shared boundary from different sources, so closure is not
    //    guaranteed.
    //
    // We apply absolute edge rounding for each element's outer box in
    // post-layout rounding to preserve closure. Border and padding widths
    // are not touched by post-layout rounding; they keep their pre-layout
    // rounded value so that they remain stable under translation.
    //
    // This gives both closure and translation stability in the case that
    // all local metrics are integer device-pixel lengths. Pre-layout
    // rounding covers that in most cases. The exception is metrics
    // resolved by layout relationships, such as percentages. Outer box
    // edges will still close globally, and painted border widths are still
    // snapped independently, but the raw content-box origin can carry a
    // 1dp residual into descendants.

    pub fn layout_bounds(&mut self, id: LayoutId, scale_factor: f32) -> Bounds<Pixels> {
        if let Some(layout) = self.absolute_layout_bounds.get(&id).cloned() {
            return layout;
        }

        let layout = self.taffy.layout(id.into()).expect(EXPECT_MESSAGE);
        let layout_location = layout.location;
        let layout_size = layout.size;
        let parent = self.taffy.parent(id.0);

        let absolute_outer_origin = match parent {
            Some(parent_id) => {
                let parent_id = LayoutId::from(parent_id);
                self.layout_bounds(parent_id, scale_factor);
                let parent_origin = *self
                    .absolute_outer_origins
                    .get(&parent_id)
                    .expect("parent absolute outer origin should be cached");
                parent_origin + Point::from(layout_location)
            }
            None => Point::from(layout_location),
        };
        self.absolute_outer_origins
            .insert(id, absolute_outer_origin);

        let absolute_far = absolute_outer_origin + Point::from(Size::from(layout_size));
        let snapped_bounds = Bounds::from_corners(
            absolute_outer_origin.map(round_half_toward_zero),
            absolute_far.map(round_half_toward_zero),
        );

        let bounds = (snapped_bounds / scale_factor).map(Pixels);
        self.absolute_layout_bounds.insert(id, bounds);
        bounds
    }
}

/// A unique identifier for a layout node, generated when requesting a layout from Taffy
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(transparent)]
pub struct LayoutId(NodeId);

impl LayoutId {
    fn to_taffy_slice(node_ids: &[Self]) -> &[taffy::NodeId] {
        // SAFETY: LayoutId is repr(transparent) to taffy::tree::NodeId.
        unsafe { std::mem::transmute::<&[LayoutId], &[taffy::NodeId]>(node_ids) }
    }
}

impl std::hash::Hash for LayoutId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        u64::from(self.0).hash(state);
    }
}

impl From<NodeId> for LayoutId {
    fn from(node_id: NodeId) -> Self {
        Self(node_id)
    }
}

impl From<LayoutId> for NodeId {
    fn from(layout_id: LayoutId) -> NodeId {
        layout_id.0
    }
}

/// Invokes a node's measure closure with taffy's device-pixel inputs, returning device pixels.
fn measure_node(
    context: &mut NodeContext,
    known_dimensions: TaffySize<Option<f32>>,
    available_space: TaffySize<TaffyAvailableSpace>,
    scale_factor: f32,
    window: &mut Window,
    cx: &mut App,
) -> TaffySize<f32> {
    let known_dimensions = Size {
        width: known_dimensions.width.map(|e| Pixels(e / scale_factor)),
        height: known_dimensions.height.map(|e| Pixels(e / scale_factor)),
    };
    let available_space: Size<AvailableSpace> = available_space.into();
    let untransform = |ev: AvailableSpace| match ev {
        AvailableSpace::Definite(pixels) => AvailableSpace::Definite(Pixels(pixels.0 / scale_factor)),
        AvailableSpace::MinContent => AvailableSpace::MinContent,
        AvailableSpace::MaxContent => AvailableSpace::MaxContent,
    };
    let available_space = size(
        untransform(available_space.width),
        untransform(available_space.height),
    );
    let measured_size: Size<Pixels> = (context.measure)(known_dimensions, available_space, window, cx);
    snap_measured_size_to_device_pixels(measured_size, scale_factor).into()
}

fn snap_measured_size_to_device_pixels(size: Size<Pixels>, scale_factor: f32) -> Size<f32> {
    size.map(|d| ceil_to_device_pixel(d.0.max(0.0), scale_factor))
}

fn border_widths_to_taffy(
    widths: &Edges<AbsoluteLength>,
    rem_size: Pixels,
    scale_factor: f32,
) -> TaffyRect<taffy::style::LengthPercentage> {
    let snap = |w: &AbsoluteLength| {
        taffy::style::LengthPercentage::length(round_stroke_to_device_pixel(
            w.to_pixels(rem_size).0,
            scale_factor,
        ))
    };
    TaffyRect {
        top: snap(&widths.top),
        right: snap(&widths.right),
        bottom: snap(&widths.bottom),
        left: snap(&widths.left),
    }
}

trait ToTaffy<Output> {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> Output;
}

impl ToTaffy<taffy::style::Style> for Style {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> taffy::style::Style {
        use taffy::style_helpers::{fr, length, minmax, repeat};

        fn to_grid_line(
            placement: &Range<crate::GridPlacement>,
        ) -> taffy::Line<taffy::GridPlacement> {
            taffy::Line {
                start: placement.start.into(),
                end: placement.end.into(),
            }
        }

        fn to_grid_repeat<T: taffy::style::CheapCloneStr>(
            unit: &Option<GridTemplate>,
        ) -> Vec<taffy::GridTemplateComponent<T>> {
            unit.map(|template| {
                match template.min_size {
                    // grid-template-*: repeat(<number>, minmax(0, 1fr));
                    crate::GridTemplateMinSize::Zero => {
                        vec![repeat(
                            template.repeat,
                            vec![minmax(length(0.0_f32), fr(1.0_f32))],
                        )]
                    }
                    // grid-template-*: repeat(<number>, minmax(min-content, 1fr));
                    crate::GridTemplateMinSize::MinContent => {
                        vec![repeat(
                            template.repeat,
                            vec![minmax(min_content(), fr(1.0_f32))],
                        )]
                    }
                    // grid-template-*: repeat(<number>, minmax(0, max-content))
                    crate::GridTemplateMinSize::MaxContent => {
                        vec![repeat(
                            template.repeat,
                            vec![minmax(length(0.0_f32), max_content())],
                        )]
                    }
                }
            })
            .unwrap_or_default()
        }

        taffy::style::Style {
            display: self.display.into(),
            overflow: self.overflow.into(),
            scrollbar_width: self.scrollbar_width.to_taffy(rem_size, scale_factor),
            position: self.position.into(),
            inset: self.inset.to_taffy(rem_size, scale_factor),
            size: self.size.to_taffy(rem_size, scale_factor),
            min_size: self.min_size.to_taffy(rem_size, scale_factor),
            max_size: self.max_size.to_taffy(rem_size, scale_factor),
            aspect_ratio: self.aspect_ratio,
            margin: self.margin.to_taffy(rem_size, scale_factor),
            padding: self.padding.to_taffy(rem_size, scale_factor),
            border: border_widths_to_taffy(&self.border_widths, rem_size, scale_factor),
            align_items: self.align_items.map(|x| x.into()),
            align_self: self.align_self.map(|x| x.into()),
            align_content: self.align_content.map(|x| x.into()),
            justify_content: self.justify_content.map(|x| x.into()),
            gap: self.gap.to_taffy(rem_size, scale_factor),
            flex_direction: self.flex_direction.into(),
            flex_wrap: self.flex_wrap.into(),
            flex_basis: self.flex_basis.to_taffy(rem_size, scale_factor),
            flex_grow: self.flex_grow,
            flex_shrink: self.flex_shrink,
            grid_template_rows: to_grid_repeat(&self.grid_rows),
            grid_template_columns: to_grid_repeat(&self.grid_cols),
            grid_row: self
                .grid_location
                .as_ref()
                .map(|location| to_grid_line(&location.row))
                .unwrap_or_default(),
            grid_column: self
                .grid_location
                .as_ref()
                .map(|location| to_grid_line(&location.column))
                .unwrap_or_default(),
            ..Default::default()
        }
    }
}

impl ToTaffy<f32> for AbsoluteLength {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> f32 {
        round_to_device_pixel(self.to_pixels(rem_size).0, scale_factor)
    }
}

impl ToTaffy<taffy::style::LengthPercentageAuto> for Length {
    fn to_taffy(
        &self,
        rem_size: Pixels,
        scale_factor: f32,
    ) -> taffy::prelude::LengthPercentageAuto {
        match self {
            Length::Definite(length) => length.to_taffy(rem_size, scale_factor),
            Length::Auto => taffy::prelude::LengthPercentageAuto::auto(),
        }
    }
}

impl ToTaffy<taffy::style::Dimension> for Length {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> taffy::prelude::Dimension {
        match self {
            Length::Definite(length) => length.to_taffy(rem_size, scale_factor),
            Length::Auto => taffy::prelude::Dimension::auto(),
        }
    }
}

impl ToTaffy<taffy::style::LengthPercentage> for DefiniteLength {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> taffy::style::LengthPercentage {
        match self {
            DefiniteLength::Absolute(length) => length.to_taffy(rem_size, scale_factor),
            DefiniteLength::Fraction(fraction) => {
                taffy::style::LengthPercentage::percent(*fraction)
            }
        }
    }
}

impl ToTaffy<taffy::style::LengthPercentageAuto> for DefiniteLength {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> taffy::style::LengthPercentageAuto {
        match self {
            DefiniteLength::Absolute(length) => length.to_taffy(rem_size, scale_factor),
            DefiniteLength::Fraction(fraction) => {
                taffy::style::LengthPercentageAuto::percent(*fraction)
            }
        }
    }
}

impl ToTaffy<taffy::style::Dimension> for DefiniteLength {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> taffy::style::Dimension {
        match self {
            DefiniteLength::Absolute(length) => length.to_taffy(rem_size, scale_factor),
            DefiniteLength::Fraction(fraction) => taffy::style::Dimension::percent(*fraction),
        }
    }
}

impl ToTaffy<taffy::style::LengthPercentage> for AbsoluteLength {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> taffy::style::LengthPercentage {
        taffy::style::LengthPercentage::length(self.to_taffy(rem_size, scale_factor))
    }
}

impl ToTaffy<taffy::style::LengthPercentageAuto> for AbsoluteLength {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> taffy::style::LengthPercentageAuto {
        taffy::style::LengthPercentageAuto::length(self.to_taffy(rem_size, scale_factor))
    }
}

impl ToTaffy<taffy::style::Dimension> for AbsoluteLength {
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> taffy::style::Dimension {
        taffy::style::Dimension::length(self.to_taffy(rem_size, scale_factor))
    }
}

impl<T, T2> From<TaffyPoint<T>> for Point<T2>
where
    T: Into<T2>,
    T2: Clone + Debug + Default + PartialEq,
{
    fn from(point: TaffyPoint<T>) -> Point<T2> {
        Point {
            x: point.x.into(),
            y: point.y.into(),
        }
    }
}

impl<T, T2> From<Point<T>> for TaffyPoint<T2>
where
    T: Into<T2> + Clone + Debug + Default + PartialEq,
{
    fn from(val: Point<T>) -> Self {
        TaffyPoint {
            x: val.x.into(),
            y: val.y.into(),
        }
    }
}

impl<T, U> ToTaffy<TaffySize<U>> for Size<T>
where
    T: ToTaffy<U> + Clone + Debug + Default + PartialEq,
{
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> TaffySize<U> {
        TaffySize {
            width: self.width.to_taffy(rem_size, scale_factor),
            height: self.height.to_taffy(rem_size, scale_factor),
        }
    }
}

impl<T, U> ToTaffy<TaffyRect<U>> for Edges<T>
where
    T: ToTaffy<U> + Clone + Debug + Default + PartialEq,
{
    fn to_taffy(&self, rem_size: Pixels, scale_factor: f32) -> TaffyRect<U> {
        TaffyRect {
            top: self.top.to_taffy(rem_size, scale_factor),
            right: self.right.to_taffy(rem_size, scale_factor),
            bottom: self.bottom.to_taffy(rem_size, scale_factor),
            left: self.left.to_taffy(rem_size, scale_factor),
        }
    }
}

impl<T, U> From<TaffySize<T>> for Size<U>
where
    T: Into<U>,
    U: Clone + Debug + Default + PartialEq,
{
    fn from(taffy_size: TaffySize<T>) -> Self {
        Size {
            width: taffy_size.width.into(),
            height: taffy_size.height.into(),
        }
    }
}

impl<T, U> From<Size<T>> for TaffySize<U>
where
    T: Into<U> + Clone + Debug + Default + PartialEq,
{
    fn from(size: Size<T>) -> Self {
        TaffySize {
            width: size.width.into(),
            height: size.height.into(),
        }
    }
}

/// The space available for an element to be laid out in
#[derive(Copy, Clone, Default, Debug, Eq, PartialEq)]
pub enum AvailableSpace {
    /// The amount of space available is the specified number of pixels
    Definite(Pixels),
    /// The amount of space available is indefinite and the node should be laid out under a min-content constraint
    #[default]
    MinContent,
    /// The amount of space available is indefinite and the node should be laid out under a max-content constraint
    MaxContent,
}

impl AvailableSpace {
    /// Returns a `Size` with both width and height set to `AvailableSpace::MinContent`.
    ///
    /// This function is useful when you want to create a `Size` with the minimum content constraints
    /// for both dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use gpui::AvailableSpace;
    /// let min_content_size = AvailableSpace::min_size();
    /// assert_eq!(min_content_size.width, AvailableSpace::MinContent);
    /// assert_eq!(min_content_size.height, AvailableSpace::MinContent);
    /// ```
    pub const fn min_size() -> Size<Self> {
        Size {
            width: Self::MinContent,
            height: Self::MinContent,
        }
    }
}

impl From<AvailableSpace> for TaffyAvailableSpace {
    fn from(space: AvailableSpace) -> TaffyAvailableSpace {
        match space {
            AvailableSpace::Definite(Pixels(value)) => TaffyAvailableSpace::Definite(value),
            AvailableSpace::MinContent => TaffyAvailableSpace::MinContent,
            AvailableSpace::MaxContent => TaffyAvailableSpace::MaxContent,
        }
    }
}

impl From<TaffyAvailableSpace> for AvailableSpace {
    fn from(space: TaffyAvailableSpace) -> AvailableSpace {
        match space {
            TaffyAvailableSpace::Definite(value) => AvailableSpace::Definite(Pixels(value)),
            TaffyAvailableSpace::MinContent => AvailableSpace::MinContent,
            TaffyAvailableSpace::MaxContent => AvailableSpace::MaxContent,
        }
    }
}

impl From<Pixels> for AvailableSpace {
    fn from(pixels: Pixels) -> Self {
        AvailableSpace::Definite(pixels)
    }
}

impl From<Size<Pixels>> for Size<AvailableSpace> {
    fn from(size: Size<Pixels>) -> Self {
        Size {
            width: AvailableSpace::Definite(size.width),
            height: AvailableSpace::Definite(size.height),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn border_widths_to_taffy_use_stroke_snapping() {
        let border_widths = Edges {
            top: Pixels(0.0).into(),
            right: Pixels(0.4).into(),
            bottom: Pixels(0.5).into(),
            left: Pixels(1.6).into(),
        };
        let taffy_border = border_widths_to_taffy(&border_widths, Pixels(16.0), 1.0);

        assert_eq!(
            taffy_border.top,
            taffy::style::LengthPercentage::length(0.0)
        );
        assert_eq!(
            taffy_border.right,
            taffy::style::LengthPercentage::length(1.0)
        );
        assert_eq!(
            taffy_border.bottom,
            taffy::style::LengthPercentage::length(1.0)
        );
        assert_eq!(
            taffy_border.left,
            taffy::style::LengthPercentage::length(2.0)
        );
    }
}

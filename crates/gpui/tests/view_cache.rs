//! Cached views (`AnyView::cached`) must behave like uncached ones: they may skip work only when
//! nothing they depend on has changed.

use std::{cell::Cell, rc::Rc};

use gpui::{
    AppContext as _, Context, Entity, Global, IntoElement, Modifiers, MouseMoveEvent,
    ParentElement, PlatformInput, Render, StyleRefinement, Styled, TestAppContext, WindowHandle, canvas, div,
    point, px,
};

/// A root that shows one cached child.
struct Root<V: Render> {
    child: Entity<V>,
}

impl<V: Render> Render for Root<V> {
    fn render(&mut self, _: &mut gpui::Window, _: &mut Context<Self>) -> impl IntoElement {
        self.child
            .clone()
            .cached(StyleRefinement::default().size(px(100.)))
    }
}

fn draw<V: Render>(window: WindowHandle<Root<V>>, cx: &mut TestAppContext) {
    cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
        .unwrap();
}

// --- replayable effects --------------------------------------------------------------------

struct EffectView {
    effect_runs: Rc<Cell<usize>>,
    prepaints: Rc<Cell<usize>>,
}

impl Render for EffectView {
    fn render(&mut self, _: &mut gpui::Window, _: &mut Context<Self>) -> impl IntoElement {
        let effect_runs = self.effect_runs.clone();
        let prepaints = self.prepaints.clone();
        canvas(
            move |_, window, cx| {
                prepaints.set(prepaints.get() + 1);
                let effect_runs = effect_runs.clone();
                window.replayable_effect(cx, move |_, _| effect_runs.set(effect_runs.get() + 1));
            },
            |_, _, _, _| {},
        )
        .size_full()
    }
}

/// State published from prepaint through `replayable_effect` must reach every frame, including
/// the ones where a cached view is reused instead of prepainted.
#[gpui::test]
fn replayable_effects_run_when_a_cached_view_is_reused(cx: &mut TestAppContext) {
    let effect_runs = Rc::new(Cell::new(0));
    let prepaints = Rc::new(Cell::new(0));
    let window = cx.add_window({
        let (effect_runs, prepaints) = (effect_runs.clone(), prepaints.clone());
        move |_, cx| Root {
            child: cx.new(|_| EffectView { effect_runs, prepaints }),
        }
    });

    // Creating the window already drew once (a real prepaint).
    assert_eq!((effect_runs.get(), prepaints.get()), (1, 1));

    draw(window, cx);
    draw(window, cx);
    assert_eq!(prepaints.get(), 1, "the view is reused, not prepainted again");
    assert_eq!(effect_runs.get(), 3, "yet its effect still runs on every frame");

    let child = window
        .read_with(cx, |root, _| root.child.clone())
        .expect("read root");
    child.update(cx, |_, cx| cx.notify());
    draw(window, cx);
    assert!(prepaints.get() > 1, "a notified view is prepainted again");

    let (effects, prepaints_before) = (effect_runs.get(), prepaints.get());
    draw(window, cx);
    assert_eq!(prepaints.get(), prepaints_before, "and is reused afterwards");
    assert_eq!(effect_runs.get(), effects + 1, "with its effect still running");
}

// --- dependencies read but not observed ----------------------------------------------------

struct Source {
    value: u32,
}

struct Reader {
    source: Entity<Source>,
    renders: Rc<Cell<usize>>,
    seen: Rc<Cell<u32>>,
}

impl Render for Reader {
    fn render(&mut self, _: &mut gpui::Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.renders.set(self.renders.get() + 1);
        self.seen.set(self.source.read(cx).value);
        div()
    }
}

/// A cached view that reads another entity without ever observing it must still re-render when
/// that entity changes; it does with an uncached view, so caching may not change it.
#[gpui::test]
fn cached_view_rerenders_when_an_entity_it_read_is_notified(cx: &mut TestAppContext) {
    let renders = Rc::new(Cell::new(0));
    let seen = Rc::new(Cell::new(0));
    let source = cx.new(|_| Source { value: 1 });
    let window = cx.add_window({
        let (source, renders, seen) = (source.clone(), renders.clone(), seen.clone());
        move |_, cx| Root {
            child: cx.new(|_| Reader { source, renders, seen }),
        }
    });
    assert_eq!((renders.get(), seen.get()), (1, 1));

    draw(window, cx);
    draw(window, cx);
    assert_eq!(renders.get(), 1, "nothing changed, so the view is reused");

    source.update(cx, |source, cx| {
        source.value = 2;
        cx.notify();
    });
    draw(window, cx);
    assert_eq!(seen.get(), 2, "the view shows the new value");

    let renders_now = renders.get();
    draw(window, cx);
    assert_eq!(renders.get(), renders_now, "and is reused again afterwards");
}

struct PaintReader {
    source: Entity<Source>,
    paints: Rc<Cell<usize>>,
    seen: Rc<Cell<u32>>,
}

impl Render for PaintReader {
    fn render(&mut self, _: &mut gpui::Window, _: &mut Context<Self>) -> impl IntoElement {
        let (source, paints, seen) = (self.source.clone(), self.paints.clone(), self.seen.clone());
        canvas(
            |_, _, _| {},
            move |_, _, _, cx| {
                paints.set(paints.get() + 1);
                seen.set(source.read(cx).value);
            },
        )
        .size_full()
    }
}

/// Reads made only while painting (a caret blink timer, say) are dependencies too.
#[gpui::test]
fn cached_view_repaints_when_an_entity_read_only_while_painting_changes(cx: &mut TestAppContext) {
    let paints = Rc::new(Cell::new(0));
    let seen = Rc::new(Cell::new(0));
    let source = cx.new(|_| Source { value: 1 });
    let window = cx.add_window({
        let (source, paints, seen) = (source.clone(), paints.clone(), seen.clone());
        move |_, cx| Root {
            child: cx.new(|_| PaintReader { source, paints, seen }),
        }
    });
    assert_eq!((paints.get(), seen.get()), (1, 1));

    draw(window, cx);
    draw(window, cx);
    assert_eq!(paints.get(), 1, "nothing changed, so the view is reused");

    source.update(cx, |source, cx| {
        source.value = 2;
        cx.notify();
    });
    draw(window, cx);
    assert_eq!(seen.get(), 2, "the view paints the new value");

    let paints_now = paints.get();
    draw(window, cx);
    assert_eq!(paints.get(), paints_now, "and is reused again afterwards");
}

// --- nesting -------------------------------------------------------------------------------

struct Outer {
    inner: Entity<Inner>,
    renders: Rc<Cell<usize>>,
}

impl Render for Outer {
    fn render(&mut self, _: &mut gpui::Window, _: &mut Context<Self>) -> impl IntoElement {
        self.renders.set(self.renders.get() + 1);
        div().child(
            self.inner
                .clone()
                .cached(StyleRefinement::default().size(px(50.))),
        )
    }
}

struct Inner {
    renders: Rc<Cell<usize>>,
}

impl Render for Inner {
    fn render(&mut self, _: &mut gpui::Window, _: &mut Context<Self>) -> impl IntoElement {
        self.renders.set(self.renders.get() + 1);
        div()
    }
}

/// A cached view that re-renders must not force the cached views nested inside it to re-render
/// too: each decides for itself.
#[gpui::test]
fn nested_cached_view_is_reused_when_its_parent_rerenders(cx: &mut TestAppContext) {
    let (outer_renders, inner_renders) = (Rc::new(Cell::new(0)), Rc::new(Cell::new(0)));
    let window = cx.add_window({
        let (outer_renders, inner_renders) = (outer_renders.clone(), inner_renders.clone());
        move |_, cx| Root {
            child: cx.new(|cx| Outer {
                inner: cx.new(|_| Inner { renders: inner_renders }),
                renders: outer_renders,
            }),
        }
    });
    assert_eq!((outer_renders.get(), inner_renders.get()), (1, 1));

    let outer = window.read_with(cx, |root, _| root.child.clone()).expect("read root");
    outer.update(cx, |_, cx| cx.notify());
    draw(window, cx);
    assert!(outer_renders.get() > 1, "the notified outer view re-renders");
    assert_eq!(inner_renders.get(), 1, "the inner view, untouched, is reused");
}

/// The ranges a cached view records for the cached views nested inside it must follow it when it
/// is replayed at a different position in the frame; otherwise a later frame that re-renders the
/// outer view and reuses the inner one replays the wrong part of the previous frame.
struct Prefixed {
    prefix_count: Rc<Cell<usize>>,
    prefix_runs: Rc<Cell<usize>>,
    outer: Entity<EffectOuter>,
}

impl Render for Prefixed {
    fn render(&mut self, _: &mut gpui::Window, _: &mut Context<Self>) -> impl IntoElement {
        let runs = self.prefix_runs.clone();
        div()
            .children((0..self.prefix_count.get()).map(move |_| {
                let runs = runs.clone();
                canvas(
                    move |_, window, cx| {
                        let runs = runs.clone();
                        window.replayable_effect(cx, move |_, _| runs.set(runs.get() + 1));
                    },
                    |_, _, _, _| {},
                )
                // Out of flow, so the cached view after them keeps its bounds.
                .absolute()
                .size(px(0.))
            }))
            .child(self.outer.clone().cached(StyleRefinement::default().size(px(60.))))
    }
}

struct EffectOuter {
    inner: Entity<EffectInner>,
}

impl Render for EffectOuter {
    fn render(&mut self, _: &mut gpui::Window, _: &mut Context<Self>) -> impl IntoElement {
        div().child(self.inner.clone().cached(StyleRefinement::default().size(px(30.))))
    }
}

struct EffectInner {
    runs: Rc<Cell<usize>>,
}

impl Render for EffectInner {
    fn render(&mut self, _: &mut gpui::Window, _: &mut Context<Self>) -> impl IntoElement {
        let runs = self.runs.clone();
        canvas(
            move |_, window, cx| {
                let runs = runs.clone();
                window.replayable_effect(cx, move |_, _| runs.set(runs.get() + 1));
            },
            |_, _, _, _| {},
        )
        .size_full()
    }
}

#[gpui::test]
fn nested_cached_view_follows_its_parent_when_the_parent_moves(cx: &mut TestAppContext) {
    let prefix_count = Rc::new(Cell::new(0));
    let (prefix_runs, inner_runs) = (Rc::new(Cell::new(0)), Rc::new(Cell::new(0)));
    let window = cx.add_window({
        let (prefix_count, prefix_runs, inner_runs) =
            (prefix_count.clone(), prefix_runs.clone(), inner_runs.clone());
        move |_, cx| Prefixed {
            prefix_count,
            prefix_runs,
            outer: cx.new(|cx| EffectOuter {
                inner: cx.new(|_| EffectInner { runs: inner_runs }),
            }),
        }
    });
    let draw = |cx: &mut TestAppContext| {
        cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx)).unwrap();
    };
    assert_eq!(inner_runs.get(), 1);

    // Frame 2: three effects now precede the outer view, which is reused at a later position.
    prefix_count.set(3);
    let root = window.read_with(cx, |root, _| root.outer.clone()).expect("read root");
    draw(cx);

    // Frame 3: the outer view re-renders and reuses the inner one from frame 2's layout.
    root.update(cx, |_, cx| cx.notify());
    let (prefix_before, inner_before) = (prefix_runs.get(), inner_runs.get());
    draw(cx);
    assert_eq!(inner_runs.get() - inner_before, 1, "the inner view's effect still runs once");
    assert_eq!(
        prefix_runs.get() - prefix_before,
        3,
        "and the prefix effects run once each, not in the inner view's place"
    );
}

// --- globals -------------------------------------------------------------------------------

struct Theme(u32);
impl Global for Theme {}

struct GlobalReader {
    renders: Rc<Cell<usize>>,
    seen: Rc<Cell<u32>>,
}

impl Render for GlobalReader {
    fn render(&mut self, _: &mut gpui::Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.renders.set(self.renders.get() + 1);
        self.seen.set(cx.global::<Theme>().0);
        div()
    }
}

/// Changing a global must reach cached views that read it, without a `refresh()`.
#[gpui::test]
fn cached_view_rerenders_when_a_global_it_read_changes(cx: &mut TestAppContext) {
    cx.set_global(Theme(1));
    let renders = Rc::new(Cell::new(0));
    let seen = Rc::new(Cell::new(0));
    let window = cx.add_window({
        let (renders, seen) = (renders.clone(), seen.clone());
        move |_, cx| Root {
            child: cx.new(|_| GlobalReader { renders, seen }),
        }
    });
    assert_eq!((renders.get(), seen.get()), (1, 1));

    draw(window, cx);
    assert_eq!(renders.get(), 1, "nothing changed, so the view is reused");

    cx.update(|cx| cx.set_global(Theme(2)));
    draw(window, cx);
    assert_eq!(seen.get(), 2, "the view shows the new global");

    let renders_now = renders.get();
    draw(window, cx);
    assert_eq!(renders.get(), renders_now, "and is reused again afterwards");
}

// --- continuously changing input -----------------------------------------------------------

struct MouseReader {
    prepaints: Rc<Cell<usize>>,
}

impl Render for MouseReader {
    fn render(&mut self, _: &mut gpui::Window, _: &mut Context<Self>) -> impl IntoElement {
        let prepaints = self.prepaints.clone();
        canvas(
            move |_, window, _| {
                prepaints.set(prepaints.get() + 1);
                let _ = window.mouse_position();
            },
            |_, _, _, _| {},
        )
        .size_full()
    }
}

/// Nothing notifies a view when the mouse moves, so a view that read the mouse position while
/// drawing is only reused while the position it saw is unchanged.
#[gpui::test]
fn cached_view_that_reads_the_mouse_follows_it(cx: &mut TestAppContext) {
    let prepaints = Rc::new(Cell::new(0));
    let window = cx.add_window({
        let prepaints = prepaints.clone();
        move |_, cx| Root {
            child: cx.new(|_| MouseReader { prepaints }),
        }
    });
    draw(window, cx);
    draw(window, cx);
    assert_eq!(prepaints.get(), 1, "the mouse is still, so the view is reused");

    window
        .update(cx, |_, window, cx| {
            window.dispatch_event(
                PlatformInput::MouseMove(MouseMoveEvent {
                    position: point(px(5.), px(7.)),
                    pressed_button: None,
                    modifiers: Modifiers::default(),
                }),
                cx,
            )
        })
        .unwrap();
    draw(window, cx);
    assert!(prepaints.get() > 1, "the mouse moved, so the view is prepainted again");

    let before = prepaints.get();
    draw(window, cx);
    assert_eq!(prepaints.get(), before, "and is reused while it stays put");
}

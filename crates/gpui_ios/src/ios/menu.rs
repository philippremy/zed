//! The iPadOS main menu, built with `UIMenuBuilder` from GPUI's platform-neutral menu model.
//!
//! `Platform::set_menus` stores the model (plus each item's key equivalent, looked up in the
//! keymap like the macOS backend does) and asks UIKit to rebuild the main menu; UIKit then calls
//! `-buildMenuWithBuilder:` on the application delegate, which calls [`build`]. Leaf items are
//! `UICommand`s / `UIKeyCommand`s that all target `-handleGPUIMenuItem:` on the delegate with the
//! item's index in the property list, so activation and enabled-state validation
//! (`-validateCommand:`) flow through the callbacks GPUI registers, the same as on macOS.
//! Items with an [`OsAction`] (cut / copy / paste / select all) target the standard responder
//! selectors instead, so the system routes them to the focused text input.

use gpui::{Action, KeyContext, Keymap, Menu, MenuItem, Modifiers, OsAction, OwnedMenu};
use objc2::{
    MainThreadMarker, Message,
    rc::Retained,
    runtime::{AnyObject, ProtocolObject, Sel},
    sel,
};
use objc2_foundation::{NSArray, NSNumber, NSString};
use objc2_ui_kit::{
    UICommand, UIKeyCommand, UIKeyModifierFlags, UIMenu, UIMenuApplication, UIMenuBuilder,
    UIMenuEdit, UIMenuElement, UIMenuElementAttributes, UIMenuElementState, UIMenuFile,
    UIMenuFormat, UIMenuHelp, UIMenuOptions, UIMenuSystem, UIMenuView, UIMenuWindow,
};
use std::{cell::RefCell, ptr::NonNull};

/// Selector every non-OS command targets; implemented by the application delegate.
pub(super) fn item_selector() -> Sel {
    sel!(handleGPUIMenuItem:)
}

/// System menus replaced wholesale by the application's own top-level menus.
fn replaced_system_menus() -> [&'static objc2_ui_kit::UIMenuIdentifier; 6] {
    // SAFETY: these are immutable framework string constants.
    unsafe {
        [
            UIMenuFile,
            UIMenuEdit,
            UIMenuFormat,
            UIMenuView,
            UIMenuWindow,
            UIMenuHelp,
        ]
    }
}

struct KeyEquivalent {
    input: String,
    modifiers: UIKeyModifierFlags,
}

enum Node {
    Separator,
    Submenu {
        title: String,
        items: Vec<Node>,
        disabled: bool,
    },
    Command {
        title: String,
        /// Index into [`Model::actions`], carried in the command's property list.
        index: usize,
        key: Option<KeyEquivalent>,
        checked: bool,
        os_selector: Option<Sel>,
    },
}

struct ActionEntry {
    action: Box<dyn Action>,
    disabled: bool,
}

#[derive(Default)]
struct Model {
    menus: Vec<Node>,
    actions: Vec<ActionEntry>,
    owned: Option<Vec<OwnedMenu>>,
}

#[derive(Default)]
struct State {
    model: Model,
    menu_command: Option<Box<dyn FnMut(&dyn Action)>>,
    validate_menu_command: Option<Box<dyn FnMut(&dyn Action) -> bool>>,
    will_open_menu: Option<Box<dyn FnMut()>>,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::default();
}

pub(super) fn set_menu_command_callback(callback: Box<dyn FnMut(&dyn Action)>) {
    STATE.with_borrow_mut(|state| state.menu_command = Some(callback));
}

pub(super) fn set_validate_callback(callback: Box<dyn FnMut(&dyn Action) -> bool>) {
    STATE.with_borrow_mut(|state| state.validate_menu_command = Some(callback));
}

#[allow(dead_code)] // UIKit has no menu-will-open hook; kept so the Platform callback has a home
pub(super) fn set_will_open_callback(callback: Box<dyn FnMut()>) {
    STATE.with_borrow_mut(|state| state.will_open_menu = Some(callback));
}

pub(super) fn owned_menus() -> Option<Vec<OwnedMenu>> {
    STATE.with_borrow(|state| state.model.owned.clone())
}

/// Store `menus` and ask UIKit to rebuild the main menu.
pub(super) fn set_menus(menus: Vec<Menu>, keymap: &Keymap) {
    let mut actions = Vec::new();
    let nodes = menus
        .iter()
        .map(|menu| Node::Submenu {
            title: menu.name.to_string(),
            items: menu.items.iter().filter_map(|item| convert(item, &mut actions, keymap)).collect(),
            disabled: menu.disabled,
        })
        .collect();
    let owned = menus.into_iter().map(Menu::owned).collect();
    STATE.with_borrow_mut(|state| {
        state.model = Model {
            menus: nodes,
            actions,
            owned: Some(owned),
        };
    });
    if let Some(mtm) = MainThreadMarker::new() {
        UIMenuSystem::mainSystem(mtm).setNeedsRebuild();
    }
}

fn convert(item: &MenuItem, actions: &mut Vec<ActionEntry>, keymap: &Keymap) -> Option<Node> {
    Some(match item {
        MenuItem::Separator => Node::Separator,
        // Menus the OS populates (macOS "Services") have no iPadOS counterpart.
        MenuItem::SystemMenu(_) => return None,
        MenuItem::Submenu(menu) => Node::Submenu {
            title: menu.name.to_string(),
            items: menu.items.iter().filter_map(|item| convert(item, actions, keymap)).collect(),
            disabled: menu.disabled,
        },
        MenuItem::Action {
            name,
            action,
            os_action,
            checked,
            disabled,
        } => {
            let os_selector = match os_action {
                Some(OsAction::Cut) => Some(sel!(cut:)),
                Some(OsAction::Copy) => Some(sel!(copy:)),
                Some(OsAction::Paste) => Some(sel!(paste:)),
                Some(OsAction::SelectAll) => Some(sel!(selectAll:)),
                // Undo / redo are application-level (there is no text-view undo stack to defer to).
                Some(OsAction::Undo | OsAction::Redo) | None => None,
            };
            let index = actions.len();
            actions.push(ActionEntry {
                action: action.boxed_clone(),
                disabled: *disabled,
            });
            Node::Command {
                title: name.to_string(),
                index,
                key: key_equivalent(action.as_ref(), keymap),
                checked: *checked,
                os_selector,
            }
        }
    })
}

/// The key equivalent to show for `action`: the first binding whose context predicate holds in a
/// default context (falling back to the first binding), as the macOS backend does. Only
/// single-keystroke bindings have one.
fn key_equivalent(action: &dyn Action, keymap: &Keymap) -> Option<KeyEquivalent> {
    let context = [KeyContext::new_with_defaults()];
    let mut first = None;
    let mut chosen = None;
    for binding in keymap.bindings_for_action(action) {
        first.get_or_insert(binding);
        if binding.predicate().is_none_or(|predicate| predicate.eval(&context)) {
            chosen = Some(binding);
            break;
        }
    }
    let keystrokes = chosen.or(first)?.keystrokes();
    let [keystroke] = keystrokes else {
        return None;
    };
    Some(KeyEquivalent {
        input: ui_key_input(keystroke.key())?,
        modifiers: ui_modifiers(keystroke.modifiers()),
    })
}

fn ui_modifiers(modifiers: &Modifiers) -> UIKeyModifierFlags {
    let mut flags = UIKeyModifierFlags::empty();
    for (held, flag) in [
        (modifiers.platform, UIKeyModifierFlags::Command),
        (modifiers.control, UIKeyModifierFlags::Control),
        (modifiers.alt, UIKeyModifierFlags::Alternate),
        (modifiers.shift, UIKeyModifierFlags::Shift),
    ] {
        if held {
            flags |= flag;
        }
    }
    flags
}

/// GPUI key name → `UIKeyCommand.input` (`UIKeyInput*` constants for the non-printing keys).
fn ui_key_input(key: &str) -> Option<String> {
    let mut chars = key.chars();
    if let (Some(only), None) = (chars.next(), chars.next()) {
        return Some(only.to_lowercase().collect());
    }
    Some(
        match key {
            "escape" => "UIKeyInputEscape",
            "up" => "UIKeyInputUpArrow",
            "down" => "UIKeyInputDownArrow",
            "left" => "UIKeyInputLeftArrow",
            "right" => "UIKeyInputRightArrow",
            "pageup" => "UIKeyInputPageUp",
            "pagedown" => "UIKeyInputPageDown",
            "home" => "UIKeyInputHome",
            "end" => "UIKeyInputEnd",
            "delete" => "UIKeyInputDelete",
            "backspace" => "\u{8}",
            "enter" => "\r",
            "tab" => "\t",
            "space" => " ",
            _ => {
                let number = key.strip_prefix('f')?.parse::<u8>().ok().filter(|n| (1..=12).contains(n))?;
                return Some(format!("UIKeyInputF{number}"));
            }
        }
        .to_owned(),
    )
}

// ---- building ------------------------------------------------------------------------------

/// Replace the system's main menu with the stored model. Called from `-buildMenuWithBuilder:`.
pub(super) fn build(builder: &ProtocolObject<dyn UIMenuBuilder>, mtm: MainThreadMarker) {
    STATE.with_borrow(|state| {
        if state.model.menus.is_empty() {
            return;
        }
        // SAFETY: framework string constants.
        let application = unsafe { UIMenuApplication };
        for identifier in replaced_system_menus() {
            builder.removeMenuForIdentifier(identifier);
        }

        for (position, node) in state.model.menus.iter().enumerate() {
            let Node::Submenu { title, items, .. } = node else {
                continue;
            };
            let elements = elements(items, mtm);
            if position == 0 {
                // The first menu is the application menu: keep the system's own title and merge
                // our items in place of its default children.
                let block = block2::RcBlock::new(move |_: NonNull<NSArray<UIMenuElement>>| {
                    NonNull::new(Retained::autorelease_ptr(elements.clone())).expect("non-null array")
                });
                // SAFETY: the block returns an autoreleased array, as the API requires.
                unsafe {
                    builder.replaceChildrenOfMenuForIdentifier_fromChildrenBlock(application, &block);
                }
            } else {
                let identifier = NSString::from_str(&format!("de.gpui.ios.menu.{position}"));
                let menu = UIMenu::menuWithTitle_image_identifier_options_children(
                    &NSString::from_str(title),
                    None,
                    Some(&identifier),
                    UIMenuOptions::empty(),
                    &elements,
                    mtm,
                );
                let previous = if position == 1 {
                    application.retain()
                } else {
                    NSString::from_str(&format!("de.gpui.ios.menu.{}", position - 1))
                };
                builder.insertSiblingMenu_afterMenuForIdentifier(&menu, &previous);
            }
        }
    });
}

/// `items` as menu elements; separators split the list into inline groups.
fn elements(items: &[Node], mtm: MainThreadMarker) -> Retained<NSArray<UIMenuElement>> {
    let mut groups: Vec<Vec<Retained<UIMenuElement>>> = vec![Vec::new()];
    for item in items {
        match item {
            Node::Separator => {
                if groups.last().is_some_and(|group| !group.is_empty()) {
                    groups.push(Vec::new());
                }
            }
            other => {
                if let Some(element) = element(other, mtm) {
                    groups.last_mut().expect("at least one group").push(element);
                }
            }
        }
    }
    groups.retain(|group| !group.is_empty());
    if groups.len() <= 1 {
        let flat: Vec<_> = groups.into_iter().flatten().collect();
        return NSArray::from_retained_slice(&flat);
    }
    let inline: Vec<Retained<UIMenuElement>> = groups
        .into_iter()
        .map(|group| {
            let menu = UIMenu::menuWithTitle_image_identifier_options_children(
                &NSString::new(),
                None,
                None,
                UIMenuOptions::DisplayInline,
                &NSArray::from_retained_slice(&group),
                mtm,
            );
            Retained::into_super(menu)
        })
        .collect();
    NSArray::from_retained_slice(&inline)
}

fn element(node: &Node, mtm: MainThreadMarker) -> Option<Retained<UIMenuElement>> {
    match node {
        Node::Separator => None,
        Node::Submenu { title, items, .. } => {
            Some(Retained::into_super(UIMenu::menuWithTitle_children(
                &NSString::from_str(title),
                &elements(items, mtm),
                mtm,
            )))
        }
        Node::Command {
            title,
            index,
            key,
            checked,
            os_selector,
        } => {
            let title = NSString::from_str(title);
            let (action, property_list) = match os_selector {
                Some(selector) => (*selector, None),
                None => (item_selector(), Some(NSNumber::new_usize(*index))),
            };
            let property_list = property_list.as_deref().map(|number| -> &AnyObject { number });
            // SAFETY: the property list is a plain NSNumber; the selectors are valid responder actions.
            let command: Retained<UICommand> = unsafe {
                match key {
                    Some(key) => Retained::into_super(
                        UIKeyCommand::commandWithTitle_image_action_input_modifierFlags_propertyList(
                            &title,
                            None,
                            action,
                            &NSString::from_str(&key.input),
                            key.modifiers,
                            property_list,
                            mtm,
                        ),
                    ),
                    None => UICommand::commandWithTitle_image_action_propertyList(
                        &title,
                        None,
                        action,
                        property_list,
                        mtm,
                    ),
                }
            };
            if *checked {
                command.setState(UIMenuElementState::On);
            }
            Some(Retained::into_super(command))
        }
    }
}

// ---- activation / validation -----------------------------------------------------------------

fn entry_index(sender: Option<&AnyObject>) -> Option<usize> {
    let sender = sender?;
    let command: &UICommand = sender.downcast_ref()?;
    let plist = command.propertyList()?;
    Some(plist.downcast_ref::<NSNumber>()?.as_usize())
}

/// `-handleGPUIMenuItem:` — run the activated item's action through GPUI's callback.
pub(super) fn perform(sender: Option<&AnyObject>) {
    let Some(index) = entry_index(sender) else {
        return;
    };
    let action = STATE.with_borrow(|state| state.model.actions.get(index).map(|e| e.action.boxed_clone()));
    let Some(action) = action else {
        return;
    };
    // Take the callback out while it runs: the action may re-enter `set_menus` or register a new one.
    let callback = STATE.with_borrow_mut(|state| state.menu_command.take());
    if let Some(mut callback) = callback {
        callback(action.as_ref());
        STATE.with_borrow_mut(|state| {
            if state.menu_command.is_none() {
                state.menu_command = Some(callback);
            }
        });
    }
}

/// `-validateCommand:` — disable the item unless it is enabled in the model and GPUI reports its
/// action as available (the same rule as the macOS `validateMenuItem:`).
pub(super) fn validate(command: &UICommand) {
    // SAFETY: reading the selector of a live command.
    if unsafe { command.action() } != item_selector() {
        return;
    }
    let plist = command.propertyList();
    let Some(index) = plist.as_deref().and_then(|p| p.downcast_ref::<NSNumber>()).map(|n| n.as_usize()) else {
        return;
    };
    let entry = STATE.with_borrow(|state| {
        state.model.actions.get(index).map(|e| (e.action.boxed_clone(), e.disabled))
    });
    let Some((action, statically_disabled)) = entry else {
        return;
    };
    let mut enabled = !statically_disabled;
    if enabled {
        let callback = STATE.with_borrow_mut(|state| state.validate_menu_command.take());
        if let Some(mut callback) = callback {
            enabled = callback(action.as_ref());
            STATE.with_borrow_mut(|state| {
                if state.validate_menu_command.is_none() {
                    state.validate_menu_command = Some(callback);
                }
            });
        }
    }
    let mut attributes = command.attributes();
    attributes.set(UIMenuElementAttributes::Disabled, !enabled);
    command.setAttributes(attributes);
}

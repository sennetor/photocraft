//! Object to Layer…: the options dialog behind the selection menu's second row. It picks how the
//! hole the object leaves is filled (a preset, or the model and its extras by hand) and takes an
//! optional prompt, as Photoshop's Generative Fill does. **Enhance** has the AI service rewrite a
//! few words into a fuller prompt (only the text is sent); it shows only when the service says it
//! can. The plain "Object to Layer" row skips the dialog and uses the Preferences defaults.
//!
//! The engine stays directly callable with parameters: this dialog only builds the params of
//! `layer.objectToLayer`.

use std::sync::{Arc, Mutex};

use egui::{Align2, Sense, pos2, vec2};
use photocraft_engine::remote;
use serde_json::{Map, Value, json};

use crate::PhotocraftApp;
use crate::theme::Tokens;

/// The shell command that opens the dialog (the engine's own command runs without one).
pub const COMMAND: &str = "layer.objectToLayerOptions";
const ENGINE_COMMAND: &str = "layer.objectToLayer";
const MARK: &str = "__objectToLayer";

/// Preset id, label, the fill method it needs, whether it improves detail, and the context in percent.
const PRESETS: [(&str, &str, &str, bool, u32); 4] =
    [("quick", "Quick fill", "fast", false, 62), ("clean", "Clean fill", "fooocus", false, 62), ("detail", "Clean fill + detail", "fooocus", true, 62), ("prompted", "Prompted fill", "fooocus", false, 50)];

pub fn owns(fields: &Map<String, Value>) -> bool {
    fields.contains_key(MARK)
}

/// The dialog's starting values: the Preferences defaults.
pub fn fields(app: &PhotocraftApp) -> Map<String, Value> {
    let i = &app.session.prefs().integrations;
    let mut f = Map::new();
    f.insert(MARK.into(), json!(true));
    f.insert("__command".into(), json!(COMMAND));
    f.insert("__label".into(), json!("Object to Layer…"));
    f.insert("method".into(), json!(i.fill_method));
    f.insert("improveDetail".into(), json!(i.fill_improve_detail));
    f.insert("context".into(), json!(i.fill_context));
    f.insert("prompt".into(), json!(""));
    f
}

pub fn open(app: &mut PhotocraftApp) -> u64 {
    // Learn which models and extras the service has now (a worker thread; the picker updates).
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = std::thread::Builder::new().name("ai-fill-methods".into()).spawn(remote::refresh_fill_methods);
    }
    let f = fields(app);
    app.ui.open_dialog(crate::state::DialogKind::Command, f)
}

/// The parameters of `layer.objectToLayer` this dialog stands for.
pub fn params(f: &Map<String, Value>) -> Value {
    let method = f.get("method").and_then(Value::as_str).unwrap_or("fast");
    let mut p = Map::new();
    p.insert("method".into(), json!(method));
    if method != "fast" {
        let prompt = f.get("prompt").and_then(Value::as_str).unwrap_or("").trim();
        if !prompt.is_empty() {
            p.insert("prompt".into(), json!(prompt.chars().take(500).collect::<String>()));
        }
        p.insert("context".into(), json!(f.get("context").and_then(Value::as_u64).unwrap_or(62).min(100) as f64 / 100.0));
        p.insert("refine".into(), json!(if f.get("improveDetail").and_then(Value::as_bool).unwrap_or(false) { 0.5 } else { 0.0 }));
    }
    Value::Object(p)
}

pub fn confirm(app: &mut PhotocraftApp, f: &Map<String, Value>) -> Result<Value, String> {
    app.run(ENGINE_COMMAND, params(f)).inspect_err(|e| {
        offer_cut(app, e);
    })
}

/// When the hole could not be filled (no service, or the fill failed), the document is unchanged:
/// say why, and offer cutting the object out without filling as an explicit choice. `true` when
/// `error` was such a failure (and is now reported).
pub fn offer_cut(app: &mut PhotocraftApp, error: &str) -> bool {
    if !error.contains(photocraft_engine::object_to_layer_cmds::NOT_FILLED) {
        return false;
    }
    let message = format!("{}: {error}", tl!("Object to Layer"));
    crate::notices::error(app, message);
    if let Some(n) = app.ui.notices.last_mut() {
        n.action = Some(crate::notices::NoticeAction { label: "Cut without filling".into(), command: ENGINE_COMMAND.into(), params: json!({"fill": false}) });
    }
    true
}

/// The preset the values match, if any.
fn current_preset(f: &Map<String, Value>) -> Option<&'static str> {
    let method = f.get("method").and_then(Value::as_str).unwrap_or("");
    let detail = f.get("improveDetail").and_then(Value::as_bool).unwrap_or(false);
    let context = f.get("context").and_then(Value::as_u64).unwrap_or(62) as u32;
    PRESETS.iter().find(|(_, _, m, d, c)| *m == method && (*m == "fast" || (*d == detail && *c == context))).map(|p| p.0)
}

fn label(ui: &mut egui::Ui, text: &str) {
    let t = Tokens::get(ui.ctx());
    let (r, _) = ui.allocate_exact_size(vec2(128.0, 22.0), Sense::hover());
    ui.painter().text(pos2(r.right() - 6.0, r.center().y), Align2::RIGHT_CENTER, text, egui::FontId::proportional(12.0), t.text_dim);
}

/// What a running "Enhance" leaves for the dialog to pick up.
type Slot = Arc<Mutex<Option<Result<String, String>>>>;

pub fn body(ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    let t = Tokens::get(ui.ctx());
    let methods = remote::fill_methods();
    ui.spacing_mut().item_spacing.y = 6.0;
    crate::widgets::section_label(ui, tl!("Fill"));
    crate::widgets::hairline(ui);

    // Presets (the ones whose model this service has), then the model by hand.
    ui.horizontal(|ui| {
        label(ui, tl!("Preset:"));
        let mut chosen = current_preset(f).unwrap_or("custom").to_string();
        let mut opts: Vec<(String, &str)> = PRESETS.iter().filter(|(_, _, m, _, _)| methods.iter().any(|x| x == m)).map(|(id, name, ..)| (id.to_string(), tl!(name))).collect();
        opts.push(("custom".into(), tl!("Custom")));
        if crate::widgets::dropdown(ui, "fill-preset", &mut chosen, &opts, 200.0)
            && let Some((_, _, m, d, c)) = PRESETS.iter().find(|p| p.0 == chosen)
        {
            f.insert("method".into(), json!(m));
            f.insert("improveDetail".into(), json!(d));
            f.insert("context".into(), json!(c));
        }
    });
    ui.horizontal(|ui| {
        label(ui, &format!("{}:", tl!("Fill method")));
        let mut method = f.get("method").and_then(Value::as_str).unwrap_or("fast").to_string();
        let labels: Vec<String> = methods.iter().map(|m| crate::prefs_ui::fill_method_label(m)).collect();
        let opts: Vec<(String, &str)> = methods.iter().cloned().zip(labels.iter().map(String::as_str)).collect();
        if crate::widgets::dropdown(ui, "fill-method", &mut method, &opts, 200.0) {
            f.insert("method".into(), json!(method));
        }
    });

    // The quick filler invents nothing, so a prompt and the extras mean nothing to it.
    let generative = f.get("method").and_then(Value::as_str).is_some_and(|m| m != "fast");
    ui.add_enabled_ui(generative, |ui| {
        ui.add_space(4.0);
        crate::widgets::section_label(ui, tl!("Prompt"));
        crate::widgets::hairline(ui);
        let mut prompt = f.get("prompt").and_then(Value::as_str).unwrap_or("").to_string();
        let slot_id = egui::Id::new("fill-enhance-slot");
        let slot: Option<Slot> = ui.ctx().data(|d| d.get_temp(slot_id));
        let finished = slot.as_ref().and_then(|s| s.lock().ok().and_then(|mut g| g.take()));
        let busy = slot.is_some() && finished.is_none();
        if let Some(done) = finished {
            ui.ctx().data_mut(|d| d.remove::<Slot>(slot_id));
            match done {
                Ok(better) => prompt = better,
                Err(e) => {
                    f.insert("__enhanceError".into(), json!(e));
                }
            }
        }
        let r = ui.add(egui::TextEdit::multiline(&mut prompt).desired_rows(3).desired_width(f32::INFINITY).char_limit(500).hint_text(tl!("Describe what to paint (optional)")));
        if r.changed() {
            f.remove("__enhanceError");
        }
        f.insert("prompt".into(), json!(prompt));
        if remote::can_enhance_prompts() {
            ui.horizontal(|ui| {
                let can = !busy && !prompt.trim().is_empty();
                if ui.add_enabled_ui(can, |ui| crate::widgets::secondary_button(ui, tl!("Enhance"), 90.0)).inner.clicked() {
                    f.remove("__enhanceError");
                    let slot: Slot = Arc::new(Mutex::new(None));
                    ui.ctx().data_mut(|d| d.insert_temp(slot_id, Arc::clone(&slot)));
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        let (text, ctx) = (prompt.clone(), ui.ctx().clone());
                        let _ = std::thread::Builder::new().name("ai-enhance-prompt".into()).spawn(move || {
                            let r = remote::enhance_prompt(&text);
                            if let Ok(mut g) = slot.lock() {
                                *g = Some(r);
                            }
                            ctx.request_repaint();
                        });
                    }
                }
                if busy {
                    ui.label(egui::RichText::new(tl!("Enhancing…")).color(t.text_dim));
                    ui.ctx().request_repaint_after(std::time::Duration::from_millis(200));
                }
            });
        }
        if let Some(e) = f.get("__enhanceError").and_then(Value::as_str) {
            ui.label(egui::RichText::new(e).color(t.danger));
        }

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            label(ui, tl!("Surrounding context"));
            let mut context = f.get("context").and_then(Value::as_u64).unwrap_or(62) as f32;
            if crate::widgets::value_field(ui, &mut context, 0.0..=100.0, "%", 84.0).changed() {
                f.insert("context".into(), json!(context.round() as u32));
            }
        });
        ui.horizontal(|ui| {
            ui.add_space(134.0);
            let mut detail = f.get("improveDetail").and_then(Value::as_bool).unwrap_or(false);
            if crate::widgets::checkbox(ui, &mut detail, tl!("Improve detail")).changed() {
                f.insert("improveDetail".into(), json!(detail));
            }
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dialog(method: &str, detail: bool, context: u32, prompt: &str) -> Map<String, Value> {
        let mut f = Map::new();
        f.insert("method".into(), json!(method));
        f.insert("improveDetail".into(), json!(detail));
        f.insert("context".into(), json!(context));
        f.insert("prompt".into(), json!(prompt));
        f
    }

    #[test]
    fn the_quick_filler_sends_only_its_method() {
        // A prompt and the extras mean nothing to a filler that invents nothing.
        assert_eq!(params(&dialog("fast", true, 30, "a road")), json!({"method": "fast"}));
    }

    #[test]
    fn a_generative_fill_sends_the_prompt_context_and_detail_pass() {
        let p = params(&dialog("fooocus", true, 50, "  empty road  "));
        assert_eq!(p, json!({"method": "fooocus", "prompt": "empty road", "context": 0.5, "refine": 0.5}));
        // No prompt: none sent (the model continues the surroundings); no detail pass: refine 0.
        let p = params(&dialog("sdxl", false, 62, "   "));
        assert_eq!(p, json!({"method": "sdxl", "context": 0.62, "refine": 0.0}));
        // Hostile values are clamped; the engine validates the rest.
        let p = params(&dialog("sdxl", false, 5000, &"x".repeat(900)));
        assert_eq!(p["context"], json!(1.0));
        assert_eq!(p["prompt"].as_str().map(|t| t.chars().count()), Some(500));
    }

    #[test]
    fn presets_are_recognised_from_their_values() {
        assert_eq!(current_preset(&dialog("fast", false, 99, "")), Some("quick"));
        assert_eq!(current_preset(&dialog("fooocus", false, 62, "")), Some("clean"));
        assert_eq!(current_preset(&dialog("fooocus", true, 62, "")), Some("detail"));
        assert_eq!(current_preset(&dialog("fooocus", false, 50, "")), Some("prompted"));
        assert_eq!(current_preset(&dialog("fooocus", false, 10, "")), None, "by hand: Custom");
        assert_eq!(current_preset(&dialog("juggernaut", false, 62, "")), None);
    }

    #[test]
    fn an_unfilled_hole_changes_nothing_and_offers_cut_without_filling() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 32, "height": 32})).unwrap();
        app.run("select.rect", json!({"x": 2, "y": 2, "width": 8, "height": 8})).unwrap();
        let ctx = egui::Context::default();
        // No AI service is set up: the command fails and the document keeps one layer.
        if remote::fill_config().is_some() {
            return;
        }
        let err = crate::menus::invoke(&mut app, &ctx, ENGINE_COMMAND, Value::Null).unwrap_err();
        assert!(err.contains("AI Integrations"), "{err}");
        assert_eq!(app.session.active().unwrap().doc.layers.len(), 1);
        let n = app.ui.notices.last().unwrap();
        assert!(n.error);
        let a = n.action.clone().unwrap();
        assert_eq!((a.label.as_str(), a.command.as_str(), &a.params), ("Cut without filling", ENGINE_COMMAND, &json!({"fill": false})));
        // The button's command is the explicit cut.
        let r = crate::menus::invoke(&mut app, &ctx, &a.command, a.params).unwrap();
        assert_eq!(r["filled"], json!(false));
        assert_eq!(app.session.active().unwrap().doc.layers.len(), 2);
        // Other failures are not mistaken for an unfilled hole.
        let before = app.ui.notices.len();
        assert!(!offer_cut(&mut app, "make a selection first"));
        assert_eq!(app.ui.notices.len(), before);
    }

    #[test]
    fn the_dialog_opens_from_the_selection_menu_with_the_preferences_defaults() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 32, "height": 32})).unwrap();
        // Without a selection there is nothing to move: the row is greyed and says why.
        assert!(!crate::menus::is_enabled(&app, COMMAND));
        app.run("select.rect", json!({"x": 2, "y": 2, "width": 8, "height": 8})).unwrap();
        assert!(crate::menus::is_enabled(&app, COMMAND));
        let ctx = egui::Context::default();
        let r = crate::menus::invoke(&mut app, &ctx, COMMAND, Value::Null).unwrap();
        let id = r["dialog"].as_u64().unwrap();
        let d = app.ui.dialogs.iter().find(|d| d.id == id).unwrap();
        assert!(owns(&d.fields));
        // Another dialog's marker must not claim it (the plain Fill dialog once did).
        assert!(!crate::fill_ui::owns(&d.fields) && !crate::pattern_fill_ui::owns(&d.fields) && !crate::stroke_ui::owns(&d.fields));
        assert_eq!(d.fields.get("method").and_then(Value::as_str), Some("fast"), "the preference's default");
        assert_eq!(d.fields.get("context").and_then(Value::as_u64), Some(62));
    }
}

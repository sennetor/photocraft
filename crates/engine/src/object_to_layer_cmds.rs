//! Object to Layer (the selection's context menu): the selected object moves onto a layer of its
//! own and the hole it leaves behind is filled with plausible background by a remote AI service
//! (an inpainting model), so the photo underneath looks as if the object had never been there.
//!
//! * The cut-out needs no service: the pixels inside the selection become a new layer above the
//!   source (what Layer via Copy makes).
//! * The fill does. The source layer is sent (about 1.6 MP, once) with the selection grown a little
//!   as the mask; the service returns the same image with the masked area filled, and only that
//!   area is blended back into the full-resolution layer. Which model fills it is a preference
//!   (`integrations.fillMethod`: `fast`, or a generative model such as `sdxl` or `juggernaut`) or
//!   the command's `method` parameter.
//! * Without a service, or when the fill fails or is cancelled, the command fails and the document
//!   is left exactly as it was: a command that promises a filled background never quietly turns
//!   into Layer via Cut. Cutting without filling is an explicit choice, `fill: false` (the UI offers
//!   it as "Cut without filling"): the hole is then left as Layer via Cut leaves it (transparent,
//!   or the background colour on the Background layer). Nothing is uploaded unless a service is
//!   set up (Preferences › AI Integrations; see [`crate::remote`]).
//!
//! The whole command is one undo step.

use std::sync::Arc;
use std::time::{Duration, Instant};

use photocraft_algo::segment::scale_for;
use photocraft_doc::{Document, Layer, LayerId};
use photocraft_geom::Rect;
use photocraft_ml::{InpaintParams, JobState, RemoteClient, RemoteConfig, method_is_valid};
use photocraft_raster::{Surface, from_rgba_into, to_rgba};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, layer_param};
use crate::decompose_cmds::{MAX_POLL_FAILURES, human_duration, poll_interval, remote_error, sample_rgba};
use crate::jobs::JobCtx;
use crate::smartselect_cmds::with_doc_sampler;
use crate::{EngineError, Result, Session};

const CMD: &str = "layer.objectToLayer";
const LABEL: &str = "Object to Layer";

/// Pixels of the image uploaded (the service works at about this size anyway).
const UPLOAD_PX: usize = 1_600_000;
/// A fill that has not finished by then is given up on (and cancelled on the server).
const DEADLINE: Duration = Duration::from_secs(10 * 60);
/// How far the hole is grown past the selection, in uploaded pixels, so halos, soft edges and
/// shadows at the object's rim are filled too.
const DEFAULT_GROW: u32 = 8;

fn bad(msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: CMD.into(), msg: msg.into() }
}

/// Why the command cannot run now.
fn enabled(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    if d.doc.selection.is_none() {
        return Err("make a selection first (the Object Selection tool selects an object with one click)".into());
    }
    let layer = d.active_layer.and_then(|id| d.doc.layer(id)).ok_or("no active layer")?;
    if layer.surface().is_none() {
        return Err("the active layer has no pixels (rasterize it first)".into());
    }
    Ok(())
}

/// The finished fill, ready to blend into the source layer.
struct Fill {
    /// The service's image, straight RGBA8, `gw × gh`.
    rgba: Vec<u8>,
    gw: usize,
    gh: usize,
    /// The hole (selection grown), 255 inside, same grid.
    hole: Vec<u8>,
    /// Document pixels per grid pixel.
    step: usize,
    window: Rect,
    model: String,
}

fn run(s: &mut Session, p: &Value) -> Result<Value> {
    let source = layer_param(s, p)?;
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let doc = Arc::clone(&d.doc);
    if doc.selection.is_none() {
        return Err(EngineError::Other("make a selection first (the Object Selection tool selects an object with one click)".into()));
    }
    let src = doc.layer(source).ok_or(EngineError::NoLayer(source))?;
    if src.surface().is_none() {
        return Err(EngineError::Other("the layer has no pixels (rasterize it first)".into()));
    }
    let source_name = src.name.clone();

    let flag = |k: &str, default: bool| p.get(k).and_then(Value::as_bool).unwrap_or(default);
    let want_fill = flag("fill", true);
    let method = match p.get("method") {
        None | Some(Value::Null) => {
            let m = crate::remote::settings().fill_method;
            if method_is_valid(&m) { m } else { "fast".to_string() }
        }
        Some(v) => v.as_str().filter(|m| method_is_valid(m)).ok_or_else(|| bad("`method` must be `fast` or the name of a generative model (letters, digits, - _ .)"))?.to_string(),
    };
    let grow = match p.get("grow") {
        None | Some(Value::Null) => DEFAULT_GROW,
        Some(v) => v.as_u64().filter(|g| *g <= 64).map(|g| g as u32).ok_or_else(|| bad("`grow` must be a whole number from 0 to 64"))?,
    };
    let prompt = match p.get("prompt") {
        None | Some(Value::Null) => String::new(),
        Some(v) => v.as_str().filter(|t| t.chars().count() <= 500).ok_or_else(|| bad("`prompt` must be text of at most 500 characters"))?.to_string(),
    };
    let seed = match p.get("seed") {
        None | Some(Value::Null) => None,
        Some(v) => Some(v.as_u64().ok_or_else(|| bad("`seed` must be a whole number"))?),
    };
    let st = crate::remote::settings();
    let unit = |k: &str, default: f32| -> Result<f32> {
        match p.get(k) {
            None | Some(Value::Null) => Ok(default),
            Some(v) => v.as_f64().filter(|x| (0.0..=1.0).contains(x)).map(|x| x as f32).ok_or_else(|| bad(format!("`{k}` must be a number from 0 to 1"))),
        }
    };
    let context = unit("context", st.fill_context as f32 / 100.0)?;
    let refine = unit("refine", if st.fill_improve_detail { 0.5 } else { 0.0 })?;
    // The extra controls are for the generative models; the quick filler ignores them.
    let params = InpaintParams { method, dilate: 0, steps: 30, prompt, seed, context: Some(context), refine: Some(refine) };

    let bg = s.tools.background;
    let apply_name = source_name.clone();
    let finish = move |s: &mut Session, fill: Option<Fill>| apply(s, fill, source, &apply_name, bg);
    if !want_fill {
        // Cut without filling, asked for explicitly: nothing to wait for.
        return finish(s, None);
    }
    let cfg = crate::remote::fill_config().ok_or_else(|| EngineError::Other(format!("{NOT_FILLED}: no AI service is set up for filling (Edit › Preferences › AI Integrations). To cut the object out anyway, use Cut without filling (`fill: false`).")))?;
    // A failed or cancelled fill fails the job, and a failed job leaves the document unchanged.
    crate::jobs::run(
        s,
        LABEL,
        true,
        move |ctx| {
            fill_via_service(ctx, &cfg, &doc, source, grow, &params).map(Some).map_err(|e| match e {
                EngineError::Cancelled => e,
                e => {
                    log::warn!("Object to Layer: the fill failed: {e}");
                    EngineError::Other(format!("{NOT_FILLED}: {e}"))
                }
            })
        },
        finish,
    )
}

/// How a failure to fill starts, so the UI can tell it from other errors and offer to cut anyway.
pub const NOT_FILLED: &str = "the hole could not be filled, so nothing was changed";

fn fill_via_service(ctx: &JobCtx, cfg: &RemoteConfig, doc: &Arc<Document>, source: LayerId, grow: u32, params: &InpaintParams) -> Result<Fill> {
    ctx.progress(0.0, "Preparing the image");
    let sel = doc.selection.as_ref().ok_or_else(|| EngineError::Other("the selection is gone".into()))?;
    let window = doc.bounds();
    let step = scale_for(window, UPLOAD_PX);
    let img = with_doc_sampler(doc, Some(source), false, |smp, _| smp.rgb_scaled(window, step));
    let (gw, gh) = (img.w, img.h);
    let mut hole = scaled_mask(sel, window, step, gw, gh);
    if !hole.contains(&255) {
        return Err(EngineError::Other("the selection is empty".into()));
    }
    dilate(&mut hole, gw, gh, grow as usize);
    let rgb: Vec<u8> = img.px.iter().flat_map(|p| p.map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8)).collect();
    ctx.check()?;
    let jpeg = photocraft_codecs::web::encode_jpeg_rgb8(gw as u32, gh as u32, &rgb, 95, false, false, None, None, None).map_err(|e| EngineError::Other(format!("could not prepare the image: {e}")))?;
    let mask_png = encode_gray_png(gw, gh, hole.clone())?;

    let client = RemoteClient::new(cfg.clone());
    ctx.progress(0.02, "Sending the image to the AI service");
    let job = client.submit_inpaint(&jpeg, "image/jpeg", &mask_png, params).map_err(remote_error)?;
    log::info!("object-to-layer fill submitted ({}, {gw}x{gh})", params.method);
    ctx.progress(0.05, "Waiting for the AI service");

    let started = Instant::now();
    let mut failures = 0;
    loop {
        if ctx.cancelled() {
            // Free the server's GPU: a cancelled job would otherwise run to the end.
            let _ = client.cancel_job(&job);
            return Err(EngineError::Cancelled);
        }
        if started.elapsed() > DEADLINE {
            let _ = client.cancel_job(&job);
            return Err(EngineError::Other("the AI service took too long (over 10 minutes); the job was cancelled".into()));
        }
        match client.job_status(&job) {
            Ok(st) => {
                failures = 0;
                match st.state {
                    JobState::Queued => {
                        let ahead = st.queue_position.map(|n| format!(" ({n} ahead)")).unwrap_or_default();
                        ctx.progress(0.05, &format!("Waiting in the AI service's queue{ahead}"));
                    }
                    JobState::Running => {
                        let frac = if st.steps > 0 { st.step as f32 / st.steps as f32 } else { 0.0 };
                        let eta = st.eta_seconds.filter(|e| *e >= 1.0).map(|e| format!(", about {} left", human_duration(e))).unwrap_or_default();
                        ctx.progress(0.1 + 0.8 * frac.clamp(0.0, 1.0), &format!("Filling the hole{eta}"));
                    }
                    JobState::Succeeded => break,
                    JobState::Failed => return Err(EngineError::Other(format!("the AI service could not fill the area{}", st.error.map(|e| format!(": {e}")).unwrap_or_default()))),
                    JobState::Cancelled | JobState::Expired => return Err(EngineError::Other("the AI service stopped the job".into())),
                }
            }
            Err(e) => {
                failures += 1;
                log::warn!("fill status check failed ({failures}/{MAX_POLL_FAILURES}): {e}");
                if failures >= MAX_POLL_FAILURES {
                    return Err(remote_error(e));
                }
            }
        }
        // Sleep in short slices so Esc is noticed quickly.
        let until = Instant::now() + poll_interval();
        while Instant::now() < until && !ctx.cancelled() {
            std::thread::sleep(Duration::from_millis(5).min(poll_interval()));
        }
    }

    ctx.progress(0.92, "Downloading the result");
    let result = client.fetch_inpainting(&job).map_err(remote_error)?;
    let _ = client.cancel_job(&job);
    ctx.check()?;
    let image = photocraft_codecs::decode(&result.png).map_err(|e| EngineError::Other(format!("AI service: the filled image could not be read: {e}")))?;
    let (fw, fh) = (image.width() as usize, image.height() as usize);
    if (fw, fh) != (gw, gh) {
        return Err(EngineError::Other(format!("AI service: the filled image is {fw}x{fh}, not the {gw}x{gh} that was sent")));
    }
    Ok(Fill { rgba: image.to_rgba8(), gw, gh, hole, step, window, model: result.model })
}

fn encode_gray_png(w: usize, h: usize, data: Vec<u8>) -> Result<Vec<u8>> {
    let img = photocraft_codecs::Image::from_u8(w as u32, h as u32, photocraft_codecs::ChannelLayout::Gray, data).map_err(|e| EngineError::Other(format!("could not prepare the mask: {e}")))?;
    photocraft_codecs::encode(&img, photocraft_codecs::Format::Png, &photocraft_codecs::EncodeOptions::default()).map_err(|e| EngineError::Other(format!("could not prepare the mask: {e}")))
}

/// The selection on the upload grid (`window` box-downsampled by `step` into `gw × gh`): 255 in
/// every cell that holds any selected pixel (even a faint one), 0 elsewhere.
fn scaled_mask(sel: &Surface, window: Rect, step: usize, gw: usize, gh: usize) -> Vec<u8> {
    let mut out = vec![0u8; gw * gh];
    let area = sel.content_bounds().intersect(&window);
    if area.is_empty() || gw == 0 || gh == 0 {
        return out;
    }
    let k = sel.format().channels().max(1);
    let w = area.width() as usize;
    let mut y0 = area.y0;
    while y0 < area.y1 {
        let y1 = (y0 + 64).min(area.y1);
        let buf = sel.read_region(Rect::new(area.x0, y0, area.x1, y1));
        for (row, line) in buf.chunks_exact(w * k).enumerate() {
            let gy = (((y0 + row as i32 - window.y0).max(0)) as usize / step).min(gh - 1);
            for (x, px) in line.chunks_exact(k).enumerate() {
                if px[0] > 0.02 {
                    let gx = (((area.x0 + x as i32 - window.x0).max(0)) as usize / step).min(gw - 1);
                    if let Some(cell) = out.get_mut(gy * gw + gx) {
                        *cell = 255;
                    }
                }
            }
        }
        y0 = y1;
    }
    out
}

/// Grows the 255 area of a `w × h` mask by `r` pixels in every direction (a square), in linear time.
fn dilate(mask: &mut [u8], w: usize, h: usize, r: usize) {
    if r == 0 || w == 0 || h == 0 || mask.len() != w * h {
        return;
    }
    // One pass along each axis: a cell is set when any cell within `r` of it on that axis is.
    let pass = |mask: &mut [u8], len: usize, count: usize, at: &dyn Fn(usize, usize) -> usize| {
        let mut prefix = vec![0u32; len + 1];
        for line in 0..count {
            for i in 0..len {
                prefix[i + 1] = prefix[i] + u32::from(mask[at(line, i)] > 0);
            }
            for i in 0..len {
                let (lo, hi) = (i.saturating_sub(r), (i + r + 1).min(len));
                if prefix[hi] > prefix[lo] {
                    mask[at(line, i)] = 255;
                }
            }
        }
    };
    // The horizontal pass reads and writes the same row; writes of 255 only add cells inside the
    // window of later cells of the same row, so it uses the prefix sums taken before writing.
    pass(mask, w, h, &|line, i| line * w + i);
    pass(mask, h, w, &|line, i| i * w + line);
}

/// Coverage (0–1) of an 8-bit grid at `(fx, fy)` in grid pixels (centres at +0.5), bilinear.
fn sample_grid(data: &[u8], w: usize, h: usize, fx: f32, fy: f32) -> f32 {
    let (fx, fy) = ((fx - 0.5).clamp(0.0, (w - 1) as f32), (fy - 0.5).clamp(0.0, (h - 1) as f32));
    let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
    let at = |x: usize, y: usize| f32::from(data.get(y * w + x).copied().unwrap_or(0));
    ((at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx) * (1.0 - ty) + (at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx) * ty) / 255.0
}

/// Blends the service's fill into `layer` inside the hole, band by band so a large layer never
/// needs one huge buffer. Outside the hole no pixel changes.
fn blend_fill(doc: &mut Document, layer: LayerId, f: &Fill) -> Result<()> {
    let surf = crate::commands::paint_surface(doc, layer, &Value::Null)?;
    let fmt = surf.format();
    let n = fmt.channels();
    // The hole's box in document pixels, one grid pixel of margin for the bilinear fringe.
    let (mut gx0, mut gy0, mut gx1, mut gy1) = (f.gw, f.gh, 0, 0);
    for (i, v) in f.hole.iter().enumerate() {
        if *v > 0 {
            let (x, y) = (i % f.gw, i / f.gw);
            (gx0, gy0, gx1, gy1) = (gx0.min(x), gy0.min(y), gx1.max(x + 1), gy1.max(y + 1));
        }
    }
    if gx1 <= gx0 || gy1 <= gy0 {
        return Ok(());
    }
    let to_doc = |g: usize, origin: i32| origin + (g * f.step) as i32;
    let area = Rect::new(to_doc(gx0.saturating_sub(1), f.window.x0), to_doc(gy0.saturating_sub(1), f.window.y0), to_doc((gx1 + 1).min(f.gw), f.window.x0), to_doc((gy1 + 1).min(f.gh), f.window.y0)).intersect(&f.window);
    if area.is_empty() {
        return Ok(());
    }
    let w = area.width() as usize;
    let inv = 1.0 / f.step as f32;
    const BAND: i32 = 128;
    let mut y = area.y0;
    while y < area.y1 {
        let yb = (y + BAND).min(area.y1);
        let mut buf = crate::allocation::read_region(surf, Rect::new(area.x0, y, area.x1, yb), "filling the hole")?;
        let blend = |(row, out): (usize, &mut [f32])| {
            let fy = ((y + row as i32 - f.window.y0) as f32 + 0.5) * inv;
            for (x, px) in out.chunks_exact_mut(n).enumerate() {
                let fx = ((area.x0 + x as i32 - f.window.x0) as f32 + 0.5) * inv;
                let c = sample_grid(&f.hole, f.gw, f.gh, fx, fy);
                if c <= 0.0 {
                    continue;
                }
                let orig = to_rgba(&fmt, px);
                let fill = sample_rgba(&f.rgba, f.gw, f.gh, fx, fy);
                let mix = |a: f32, b: f32| a + (b - a) * c;
                from_rgba_into(&fmt, [mix(orig[0], fill[0]), mix(orig[1], fill[1]), mix(orig[2], fill[2]), mix(orig[3], 1.0)], px);
            }
        };
        {
            use rayon::prelude::*;
            buf.par_chunks_mut(w * n).enumerate().for_each(blend);
        }
        surf.write_region(Rect::new(area.x0, y, area.x1, yb), &buf);
        y = yb;
    }
    surf.prune();
    Ok(())
}

/// Cuts the object onto its own layer above the source and fills the hole (or, for an explicit
/// cut without filling, clears it), as one undo step.
fn apply(s: &mut Session, fill: Option<Fill>, source: LayerId, source_name: &str, background: [f32; 4]) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let sel = d.doc.selection.as_ref().ok_or_else(|| EngineError::Other("the selection is gone".into()))?;
    let surf = d.doc.layer(source).and_then(Layer::surface).ok_or(EngineError::NoLayer(source))?;
    let object = crate::edit_cmds::lift_selection(surf, sel, d.doc.bounds())?;
    let (filled, model) = (fill.is_some(), fill.as_ref().map(|f| f.model.clone()));
    let nid = s.edit(LABEL, move |doc, active| {
        match &fill {
            Some(f) => blend_fill(doc, source, f)?,
            None => crate::edit_cmds::clear_selected(doc, source, background)?,
        }
        let mut l = Layer::raster(doc.next_layer_name("Object"), object.format());
        *crate::pixels_mut(&mut l)? = object;
        let nid = doc.insert_above(Some(source), l);
        *active = Some(nid);
        // As Layer via Cut does: deselect, so a following Free Transform moves the whole object.
        doc.selection = None;
        Ok(nid)
    })?;
    log::info!("object to layer from \"{source_name}\": filled={filled}");
    Ok(json!({ "layer": nid.0, "filled": filled, "model": model }))
}

/// Object to Layer command spec (the selection's context menu).
pub fn specs() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: LABEL,
        menu: &[],
        shortcut: None,
        params: r##"{"layer":id?,"method":"fast"|<generative model id, e.g. sdxl, juggernaut>=prefs,"fill":bool=true,"grow":0..64=8,"prompt":str?,"context":0..1=prefs,"refine":0..1=prefs,"seed":int?} → {layer,filled,model} (cuts the selected pixels onto a new layer above the source and fills the hole through the AI service chosen in Preferences › AI Integrations; with no service, or if the fill fails or is cancelled, it fails and the document is unchanged; `fill: false` cuts without filling, leaving the hole as Layer via Cut does)"##,
        enabled,
        run,
        journal: true,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    fn png_rgba(w: u32, h: u32, px: [u8; 4]) -> Vec<u8> {
        let data: Vec<u8> = (0..w * h).flat_map(|_| px).collect();
        let img = photocraft_codecs::Image::from_u8(w, h, photocraft_codecs::ChannelLayout::Rgba, data).unwrap();
        photocraft_codecs::encode(&img, photocraft_codecs::Format::Png, &photocraft_codecs::EncodeOptions::default()).unwrap()
    }

    fn fill_zip(w: u32, h: u32) -> Vec<u8> {
        let mut z = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        let manifest = format!(r#"{{"kind":"inpaint","model":"big-lama","mode":"fast","input":{{"width":{w},"height":{h}}}}}"#);
        for (name, data) in [("manifest.json", manifest.into_bytes()), ("fill.png", png_rgba(w, h, [0, 255, 0, 255]))] {
            z.start_file(name, zip::write::SimpleFileOptions::default()).unwrap();
            z.write_all(&data).unwrap();
        }
        z.finish().unwrap().into_inner()
    }

    struct Reply(&'static str, &'static str, Vec<u8>);

    fn json_reply(status: &'static str, body: &str) -> Reply {
        Reply(status, "application/json", body.as_bytes().to_vec())
    }

    /// Waits (up to 10 s) for the next connection and reads its request; returns the stream and
    /// the request line, or `None` when nobody connected.
    fn next_request(l: &TcpListener) -> Option<(std::net::TcpStream, String)> {
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut s = loop {
            match l.accept() {
                Ok((s, _)) => break s,
                Err(_) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(5)),
                Err(_) => return None,
            }
        };
        s.set_nonblocking(false).unwrap();
        let mut got = Vec::new();
        let mut buf = [0u8; 8192];
        let head_end = loop {
            if let Some(p) = got.windows(4).position(|w| w == b"\r\n\r\n") {
                break p + 4;
            }
            let n = s.read(&mut buf).unwrap();
            if n == 0 {
                break got.len();
            }
            got.extend(&buf[..n]);
        };
        let head = String::from_utf8_lossy(&got[..head_end]).to_string();
        let want = head.to_ascii_lowercase().lines().find_map(|l| l.strip_prefix("content-length:").and_then(|v| v.trim().parse::<usize>().ok())).unwrap_or(0);
        while got.len() < head_end + want {
            let n = s.read(&mut buf).unwrap();
            if n == 0 {
                break;
            }
            got.extend(&buf[..n]);
        }
        Some((s, head.lines().next().unwrap_or("").to_string()))
    }

    fn reply(s: &mut std::net::TcpStream, Reply(status, ct, body): Reply) {
        write!(s, "HTTP/1.1 {status}\r\nContent-Type: {ct}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).unwrap();
        s.write_all(&body).unwrap();
    }

    fn listen() -> (TcpListener, String) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        l.set_nonblocking(true).unwrap();
        let url = format!("http://{}/photocraft/v1", l.local_addr().unwrap());
        (l, url)
    }

    /// A loopback server answering the next requests in order; returns each request line.
    fn serve(replies: Vec<Reply>) -> (String, std::thread::JoinHandle<Vec<String>>) {
        let (l, url) = listen();
        let h = std::thread::spawn(move || {
            let mut seen = Vec::new();
            for r in replies {
                let Some((mut s, line)) = next_request(&l) else { return seen };
                reply(&mut s, r);
                seen.push(line);
            }
            seen
        });
        (url, h)
    }

    /// A loopback service whose fill job never finishes: it accepts the job and answers "running"
    /// until the job is cancelled (a DELETE); returns each request line.
    fn serve_until_cancelled() -> (String, std::thread::JoinHandle<Vec<String>>) {
        let (l, url) = listen();
        let h = std::thread::spawn(move || {
            let mut seen = Vec::new();
            while let Some((mut s, line)) = next_request(&l) {
                let done = line.starts_with("DELETE");
                let r = if line.starts_with("POST") {
                    json_reply("202 Accepted", r#"{"job_id":"01JFILL","state":"queued"}"#)
                } else if done {
                    json_reply("204 No Content", "")
                } else {
                    json_reply("200 OK", r#"{"state":"running","step":1,"steps":30}"#)
                };
                reply(&mut s, r);
                seen.push(line);
                if done {
                    break;
                }
            }
            seen
        });
        (url, h)
    }

    /// What must not change when the command does not run: the document snapshot itself, its
    /// undo history and its selection.
    fn assert_unchanged(s: &Session, before: &Arc<Document>, steps: usize) {
        let st = s.active().unwrap();
        assert!(Arc::ptr_eq(&st.doc, before), "the document was replaced");
        assert_eq!(st.history.past_len(), steps, "no undo step was added");
        assert_eq!(st.doc.layers.len(), 1, "no object layer");
        assert!(st.doc.selection.is_some(), "the selection is kept, ready for another try");
        assert!(px(s, 0, 32, 24)[2] > 0.85, "the object is still in the photo");
    }

    fn snapshot(s: &Session) -> (Arc<Document>, usize) {
        let st = s.active().unwrap();
        (Arc::clone(&st.doc), st.history.past_len())
    }

    fn configure(url: &str) {
        crate::remote::set_test_config(photocraft_ml::RemoteConfig::new(url, "pc_test_key"));
    }

    /// A 64 × 48 photo: grey, with a blue 12 × 12 "object" at (26, 18), and that object selected.
    fn session_with_object() -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 64, "height": 48})).unwrap();
        s.edit("paint", |doc, a| {
            let surf = doc.layer_mut(a.unwrap()).unwrap().surface_mut().unwrap();
            surf.fill_rect(Rect::new(0, 0, 64, 48), &[0.5, 0.5, 0.5, 1.0]);
            surf.fill_rect(Rect::new(26, 18, 38, 30), &[0.1, 0.1, 0.9, 1.0]);
            Ok(())
        })
        .unwrap();
        s.execute("select.rect", json!({"x": 26, "y": 18, "width": 12, "height": 12})).unwrap();
        s
    }

    fn px(s: &Session, layer: usize, x: i32, y: i32) -> [f32; 4] {
        let mut o = [[0.0f32; 4]; 1];
        s.active().unwrap().doc.layers[layer].surface().unwrap().read_rgba_into(Rect::new(x, y, x + 1, y + 1), &mut o);
        o[0]
    }

    fn happy_replies() -> Vec<Reply> {
        vec![
            json_reply("202 Accepted", r#"{"job_id":"01JFILL","state":"queued"}"#),
            json_reply("200 OK", r#"{"state":"running","step":1,"steps":2,"eta_seconds":1}"#),
            json_reply("200 OK", r#"{"state":"succeeded","step":2,"steps":2}"#),
            Reply("200 OK", "application/zip", fill_zip(64, 48)),
            json_reply("204 No Content", ""),
        ]
    }

    #[test]
    fn the_object_gets_its_own_layer_and_the_hole_is_filled_inside_the_grown_mask_only() {
        let (url, h) = serve(happy_replies());
        configure(&url);
        let mut s = session_with_object();
        let steps = s.active().unwrap().history.past_len();
        let r = s.execute(CMD, json!({"method": "fast"})).unwrap();
        assert_eq!((r["filled"].as_bool(), r["model"].as_str()), (Some(true), Some("big-lama")), "{r}");
        let st = s.active().unwrap();
        assert_eq!(st.doc.layers.len(), 2, "background and the object");
        assert!(st.doc.selection.is_none(), "deselected, like Layer via Cut");
        assert_eq!(st.history.past_len(), steps + 1, "one undo step");
        // The object layer holds the blue pixels and nothing around them.
        let o = px(&s, 1, 30, 22);
        assert!(o[2] > 0.85 && o[3] > 0.99, "{o:?}");
        assert!(px(&s, 1, 5, 5)[3] < 0.01);
        // The hole (selection + 8 px) is green, beyond it the grey photo is untouched.
        let inside = px(&s, 0, 32, 24);
        assert!(inside[1] > 0.9 && inside[0] < 0.1, "{inside:?}");
        let grown = px(&s, 0, 22, 24);
        assert!(grown[1] > 0.9, "within the grown mask: {grown:?}");
        let outside = px(&s, 0, 10, 24);
        assert!((outside[0] - 0.5).abs() < 0.01 && (outside[1] - 0.5).abs() < 0.01, "{outside:?}");
        let reqs = h.join().unwrap();
        assert!(reqs[0].starts_with("POST /photocraft/v1/inpaint") && reqs[3].contains("/result") && reqs[4].starts_with("DELETE"), "{reqs:?}");
        // One undo brings the photo and selection back.
        s.execute("edit.undo", json!({})).unwrap();
        let st = s.active().unwrap();
        assert_eq!(st.doc.layers.len(), 1);
        assert!(st.doc.selection.is_some());
        let again = px(&s, 0, 32, 24);
        assert!(again[2] > 0.85, "the object is back: {again:?}");
        crate::remote::set_test_config(None);
    }

    #[test]
    fn a_failing_service_changes_nothing_and_says_why() {
        let (url, h) = serve(vec![json_reply("401 Unauthorized", r#"{"title":"Invalid API key","status":401}"#)]);
        configure(&url);
        let mut s = session_with_object();
        let (before, steps) = snapshot(&s);
        let why = s.execute(CMD, json!({})).unwrap_err().to_string();
        assert!(why.contains(NOT_FILLED) && why.contains("401") && !why.contains("pc_test_key"), "{why}");
        assert_unchanged(&s, &before, steps);
        h.join().unwrap();
        crate::remote::set_test_config(None);
    }

    #[test]
    fn a_fill_job_the_service_fails_changes_nothing() {
        let (url, h) = serve(vec![
            json_reply("202 Accepted", r#"{"job_id":"01JFILL","state":"queued"}"#),
            json_reply("200 OK", r#"{"state":"failed","error":"out of GPU memory"}"#),
        ]);
        configure(&url);
        let mut s = session_with_object();
        let (before, steps) = snapshot(&s);
        let why = s.execute(CMD, json!({})).unwrap_err().to_string();
        assert!(why.contains(NOT_FILLED) && why.contains("out of GPU memory"), "{why}");
        assert_unchanged(&s, &before, steps);
        h.join().unwrap();
        crate::remote::set_test_config(None);
    }

    #[test]
    fn without_a_service_it_changes_nothing_and_points_at_the_settings_and_the_explicit_cut() {
        crate::remote::set_test_config(None);
        let mut s = session_with_object();
        let (before, steps) = snapshot(&s);
        let why = s.execute(CMD, json!({})).unwrap_err().to_string();
        assert!(why.contains(NOT_FILLED) && why.contains("AI Integrations") && why.contains("fill: false"), "{why}");
        assert_unchanged(&s, &before, steps);
    }

    #[test]
    fn cancelling_the_fill_changes_nothing_and_cancels_the_server_job() {
        let (url, h) = serve_until_cancelled();
        configure(&url);
        let mut s = session_with_object();
        let (before, steps) = snapshot(&s);
        let id = match s.start(CMD, json!({})).unwrap() {
            crate::jobs::Started::Job(id) => id,
            crate::jobs::Started::Done(v) => panic!("expected a background job: {v}"),
        };
        // Cancel once the job is with the service, as Esc does while it fills.
        let t = Instant::now();
        while s.job(id).is_some_and(|j| j.progress < 0.05) {
            assert!(t.elapsed() < Duration::from_secs(30), "the job never reached the service");
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!(s.cancel_job(id));
        s.join_cancelled_jobs();
        assert!(matches!(s.wait_job(id), Err(EngineError::Cancelled)));
        assert!(s.poll_jobs().iter().all(|e| e.id != id || e.outcome == crate::jobs::JobOutcome::Cancelled));
        assert_unchanged(&s, &before, steps);
        let reqs = h.join().unwrap();
        assert!(reqs.last().is_some_and(|r| r.starts_with("DELETE")), "the server's GPU is freed: {reqs:?}");
        crate::remote::set_test_config(None);
    }

    #[test]
    fn cut_without_filling_is_the_explicit_choice_and_works_like_layer_via_cut() {
        // `fill: false` skips the service even when one is set up (nothing listens on port 1).
        configure("http://127.0.0.1:1/photocraft/v1");
        let mut s = session_with_object();
        let steps = s.active().unwrap().history.past_len();
        let r = s.execute(CMD, json!({"fill": false})).unwrap();
        assert_eq!((r["filled"].as_bool(), r.get("warning")), (Some(false), None), "{r}");
        let st = s.active().unwrap();
        assert_eq!((st.doc.layers.len(), st.history.past_len()), (2, steps + 1));
        assert!(px(&s, 1, 30, 22)[2] > 0.85, "the object is on its layer");
        // The Background cannot hold a hole: it takes the background colour (white by default).
        let hole = px(&s, 0, 32, 24);
        assert!(hole[0] > 0.9 && hole[1] > 0.9 && hole[2] > 0.9, "{hole:?}");
        crate::remote::set_test_config(None);
    }

    #[test]
    fn it_needs_a_selection_and_a_pixel_layer_and_validates_params() {
        configure("http://127.0.0.1:1/photocraft/v1");
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 32, "height": 32})).unwrap();
        assert!(enabled(&s).is_err());
        assert!(s.execute(CMD, json!({})).unwrap_err().to_string().contains("selection"));
        let mut s = session_with_object();
        assert!(enabled(&s).is_ok());
        for p in [json!({"method": "a b"}), json!({"method": 5}), json!({"grow": 65}), json!({"prompt": 5}), json!({"seed": -1}), json!({"context": 1.5}), json!({"refine": "x"})] {
            assert!(s.execute(CMD, p.clone()).is_err(), "{p}");
        }
        crate::remote::set_test_config(None);
    }

    #[test]
    fn the_mask_covers_any_selected_cell_and_grows_by_the_radius() {
        let mut sel = Surface::new(photocraft_color::PixelFormat::GRAY8);
        sel.fill_rect(Rect::new(10, 10, 14, 14), &[1.0]);
        sel.fill_rect(Rect::new(20, 10, 21, 11), &[0.1]);
        let m = scaled_mask(&sel, Rect::new(0, 0, 40, 40), 2, 20, 20);
        let at = |x: usize, y: usize| m[y * 20 + x];
        assert_eq!((at(5, 5), at(6, 6), at(4, 5), at(7, 7)), (255, 255, 0, 0), "a 4x4 selection is 2x2 cells");
        assert_eq!(at(10, 5), 255, "a faint pixel still counts");
        let mut d = m.clone();
        dilate(&mut d, 20, 20, 2);
        assert_eq!((d[5 * 20 + 3], d[5 * 20 + 2], d[3 * 20 + 5], d[8 * 20 + 8], d[9 * 20 + 9]), (255, 0, 255, 255, 0));
        // Radius 0 and mismatched sizes change nothing.
        let mut same = m.clone();
        dilate(&mut same, 20, 20, 0);
        dilate(&mut same, 7, 7, 3);
        assert_eq!(same, m);
    }

    /// Selects the object at a point in a photo, runs Object to Layer against the real service with
    /// each fill method, and dumps the background layer alone (`<method>.raw`: width, height as
    /// little-endian u32, then RGB8) for viewing.
    /// `PHOTOCRAFT_REMOTE_TEST=1 PHOTOCRAFT_REMOTE_URL=… PHOTOCRAFT_REMOTE_KEY=… PHOTOCRAFT_TEST_IMAGE=photo.jpg
    /// PHOTOCRAFT_TEST_POINT=x,y PHOTOCRAFT_TEST_OUT=dir [PHOTOCRAFT_TEST_METHODS=fast,sdxl,juggernaut]
    /// cargo test -p photocraft-engine --release --lib object_to_layer_live -- --ignored --nocapture`
    #[test]
    #[ignore = "needs a remote AI service and a photo"]
    fn object_to_layer_live_dump() {
        let (Ok(img), Ok(pt), Ok(out)) = (std::env::var("PHOTOCRAFT_TEST_IMAGE"), std::env::var("PHOTOCRAFT_TEST_POINT"), std::env::var("PHOTOCRAFT_TEST_OUT")) else { return };
        let xy: Vec<f64> = pt.split(',').filter_map(|v| v.trim().parse().ok()).collect();
        let methods = std::env::var("PHOTOCRAFT_TEST_METHODS").unwrap_or_else(|_| "fast,sdxl,juggernaut".into());
        std::fs::create_dir_all(&out).unwrap();
        for method in methods.split(',') {
            let mut s = Session::new();
            s.execute("file.openAs", json!({"path": img})).unwrap();
            s.execute("select.object", json!({"point": xy})).unwrap();
            let mut params = json!({"method": method, "seed": 1});
            if let Ok(prompt) = std::env::var("PHOTOCRAFT_TEST_PROMPT") {
                // PHOTOCRAFT_TEST_ENHANCE=1 first improves the prompt through the language model.
                let prompt = if std::env::var("PHOTOCRAFT_TEST_ENHANCE").as_deref() == Ok("1") { crate::remote::enhance_prompt(&prompt).unwrap() } else { prompt };
                eprintln!("prompt: {prompt}");
                params["prompt"] = json!(prompt);
            }
            for (env, key) in [("PHOTOCRAFT_TEST_REFINE", "refine"), ("PHOTOCRAFT_TEST_CONTEXT", "context")] {
                if let Some(v) = std::env::var(env).ok().and_then(|v| v.parse::<f64>().ok()) {
                    params[key] = json!(v);
                }
            }
            let t = Instant::now();
            let r = s.execute(CMD, params).unwrap();
            eprintln!("{method}: {r} in {:.1} s", t.elapsed().as_secs_f32());
            let doc = &s.active().unwrap().doc;
            let b = doc.bounds();
            let mut solo = (**doc).clone();
            for (i, l) in solo.layers.iter_mut().enumerate() {
                l.visible = i == 0;
            }
            let px = photocraft_compose::render(&solo, b).px;
            let mut bytes = Vec::new();
            bytes.extend(b.width().to_le_bytes());
            bytes.extend(b.height().to_le_bytes());
            for p in &px {
                bytes.extend(p[..3].iter().map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8));
            }
            let tag = std::env::var("PHOTOCRAFT_TEST_TAG").unwrap_or_default();
            std::fs::write(std::path::Path::new(&out).join(format!("{method}{tag}.raw")), bytes).unwrap();
        }
    }

    #[test]
    fn the_menu_item_ids_are_live_commands() {
        assert!(crate::commands::find(CMD).is_some());
    }
}

// Builds docs/presentation/StructRender_Lab_Presentation.pptx: white background, maroon 48 pt headings,
// black content, Times New Roman. Every number is taken from the lab report (docs/lab_report), which was
// checked against the committed evaluation reports.
// Run: NODE_PATH=<dir with pptxgenjs> node docs/presentation/build/build_deck.js <out.pptx>
"use strict";
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const ROOT = path.resolve(__dirname, "../../..");
const LAB = (f) => path.join(ROOT, "docs/lab_report/figures", f);
const PAPER = (f) => path.join(ROOT, "docs/paper/figures", f);
const OUT = process.argv[2];

const THEME = {
  name: "StructRender Maroon",
  headFontFace: "Times New Roman",
  bodyFontFace: "Times New Roman",
  colors: { dk1: "000000", lt1: "FFFFFF", dk2: "800000", lt2: "F6EEEE", accent1: "800000", accent2: "B5651D",
    accent3: "595959", accent4: "C9A3A3", accent5: "404040", accent6: "8C8C8C", hlink: "800000", folHlink: "5C0000" },
};
const MAROON = "800000", INK = "000000", GREY = "595959";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.title = "StructRender: Engine-Grounded Real-Time Neural Rendering of a Game in Learned Painterly Styles";
pres.author = "Kabilash S, Pavitra P";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ------------------------------------------------------------------ layouts
pres.defineSlideMaster({ title: "TITLE", background: { color: "FFFFFF" } });
pres.defineSlideMaster({
  title: "CONTENT", background: { color: "FFFFFF" },
  margin: [0.5, 0.5, 0.7, 0.5],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.55, y: 0.3, w: 12.2, h: 0.95, fontSize: 48, bold: true,
      color: C.text2, fontFace: "Times New Roman", align: "left", valign: "middle", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 0.55, y: 6.98, w: 0.6, h: 0.35, fontSize: 12, color: GREY, fontFace: "Times New Roman" },
});

// ------------------------------------------------------------------ helpers
let section = "";
function slide(title, notes, sec) {
  if (sec && sec !== section) { pres.addSection({ title: sec }); section = sec; }
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  return s;
}
function bullets(s, items, o) {
  s.addText(items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } })),
    { isTextBox: true, fontSize: 20, color: INK, valign: "top", paraSpaceAfter: 8, margin: 0.05, ...o });
}
function leadBullets(s, items, o) { // [lead, text] pairs: bold lead, plain text
  const runs = [];
  items.forEach(([h, b], i) => {
    runs.push({ text: h + ": ", options: { bullet: true, bold: true } });
    runs.push({ text: b, options: { breakLine: i < items.length - 1 } });
  });
  s.addText(runs, { isTextBox: true, fontSize: 20, color: INK, valign: "top", paraSpaceAfter: 10, margin: 0.05, ...o });
}
const STAGES = ["Godot game", "Export", "Record", "Teacher", "Gates", "Train", "Deploy", "Live"];
function pipeline(s, from, to) { // the whole pipeline on one line; this slide's stages in maroon
  s.addText(STAGES.flatMap((t, i) => [
    { text: t, options: i >= from && i <= to ? { bold: true, color: MAROON } : { color: "8C8C8C" } },
    ...(i < STAGES.length - 1 ? [{ text: "  →  ", options: { color: "8C8C8C" } }] : [])]),
  { isTextBox: true, x: 0.55, y: 1.3, w: 12.2, h: 0.45, fontSize: 18, margin: 0, valign: "middle", objectName: "pipeline" });
}
function caption(s, text, x, y, w, h = 0.4) {
  s.addText(text, { isTextBox: true, x, y, w, h, fontSize: 14, italic: true, color: GREY, align: "left", margin: 0, valign: "top" });
}
function img(s, file, x, y, w, aspect, alt) { // aspect = width / height of the source image
  s.addImage({ path: file, x, y, w, h: w / aspect, altText: alt });
}

// ================================================================== 1. Title
pres.addSection({ title: "Title" }); section = "Title";
{
  const s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Title" });
  s.addImage({ path: LAB("sriher_logo.png"), x: 2.67, y: 0.35, w: 8.0, h: 1.6, altText: "Sri Ramachandra Institute of Higher Education and Research logo" });
  s.addText("ADA23DLU06 – RECENT TRENDS IN DEEP LEARNING LABORATORY", { isTextBox: true, x: 0.6, y: 2.15, w: 12.13, h: 0.5,
    fontSize: 22, bold: true, color: INK, align: "center", margin: 0 });
  s.addText("StructRender: Engine-Grounded Real-Time Neural Rendering of a Game in Learned Painterly Styles",
    { isTextBox: true, x: 0.9, y: 2.85, w: 11.53, h: 1.7, fontSize: 40, bold: true, color: MAROON, align: "center", valign: "middle", margin: 0 });
  s.addText([
    { text: "Submitted by", options: { italic: true, bold: true, fontSize: 18, breakLine: true } },
    { text: "Kabilash S – E0323036", options: { bold: true, fontSize: 20, breakLine: true } },
    { text: "Pavitra P – E0323038", options: { bold: true, fontSize: 20 } },
  ], { isTextBox: true, x: 0.6, y: 5.35, w: 5.5, h: 1.4, color: INK, align: "left", valign: "bottom", margin: 0 });
  s.addText([
    { text: "Under the guidance of", options: { italic: true, bold: true, fontSize: 18, breakLine: true } },
    { text: "Dr. V. Vanitha", options: { bold: true, fontSize: 20, breakLine: true } },
    { text: "Associate Professor, Dept. of CSE (AIML)", options: { fontSize: 18 } },
  ], { isTextBox: true, x: 7.23, y: 5.35, w: 5.5, h: 1.4, color: INK, align: "right", valign: "bottom", margin: 0 });
  s.addNotes("StructRender: a game engine keeps all gameplay truth, and a small neural network renders each frame in a learned painted style, live.");
}

// ================================================================== 2. Problem statement
{
  const s = slide("Problem Statement", "State the problem, then the obstacles that make it hard.", "Introduction");
  s.addText("Render the structure a game engine already computes for each frame in a learned artistic style, at interactive frame rates, while the engine, not the network, decides what exists and where.",
    { isTextBox: true, x: 0.55, y: 1.5, w: 12.2, h: 1.2, fontSize: 24, color: INK, margin: 0, valign: "top" });
  leadBullets(s, [
    ["World facts drift", "generative game models such as GameNGen and Genie [10], [11] predict pixels and hold no authoritative game state."],
    ["No paired data", "painted versions of game frames do not exist, and a diffusion teacher can invent geometry."],
    ["Flicker", "styling each frame on its own makes the output shimmer from frame to frame [9]."],
    ["Real-time limits", "the renderer must run live on an 8 GB laptop GPU and must provably be the model that was evaluated."],
  ], { x: 0.55, y: 2.95, w: 12.2, h: 3.8 });
}

// ================================================================== 3. Introduction
{
  const s = slide("Introduction", "Neural rendering from engine buffers; StructRender keeps the engine in charge.");
  bullets(s, [
    "Changing a game's look, for example into an oil painting, normally means re-authoring its materials and shaders.",
    "Neural rendering lets a network map scene information to an image in a target appearance [1].",
    "Game engines already compute that information every frame: depth, normals and albedo, the G-buffer [2].",
    "Generative world models instead predict whole frames from pixels, so objects and positions can drift [10], [11].",
    "StructRender keeps Godot in charge of physics, logic and entity identity; the network decides only how each frame looks.",
    "A diffusion model is used offline only, to make training targets; a small U-Net runs live through TensorRT.",
    "The style can be switched during play without changing anything in the game.",
  ], { x: 0.55, y: 1.45, w: 12.2, h: 3.3, fontSize: 18, paraSpaceAfter: 5 });
  img(s, PAPER("teaser.png"), 1.4, 4.8, 10.5, 2166 / 449, "Native Godot frame and the same frame rendered live in oil and watercolour");
  caption(s, "The same live frame: native Godot render, StructRender oil, StructRender watercolour.", 1.4, 6.98, 9.0, 0.35);
}

// ================================================================== 4. Objectives
{
  const s = slide("Objectives", "Eight objectives, each with a measurable target fixed before the results existed.");
  bullets(s, [
    "Export depth, normals, albedo, motion and entity IDs from the Godot game every frame.",
    "Generate styled training targets with a diffusion teacher, accepted only through 4 alignment gates.",
    "Reach sharpness ≥ 0.90 and colour travel ≥ 0.80 relative to the teacher on held-out sessions.",
    "Reduce flicker with engine-guided temporal reprojection beyond a pre-registered noise threshold.",
    "Compare U-Net and transformer renderers, temporal input, entity awareness and memory at equal size.",
    "Render two styles with one model and switch between them during play.",
    "Deploy through ONNX, TensorRT and C++ in Godot at about 30 fps, with a native-frame fallback.",
    "Test whether a per-entity appearance memory (EPAM) improves the output.",
  ], { x: 0.55, y: 1.5, w: 12.2, h: 5.3, fontSize: 20, paraSpaceAfter: 9 });
}

// ================================================================== 5. Survey
{
  const s = slide("Basic Research Survey", "Related work in four groups; the last column is what each leaves open for this project.", "Background");
  const head = ["Work", "Method", "Gap for this project"].map((t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: MAROON } } }));
  const rows = [
    ["Deep Shading, 2017 [3]", "CNN maps G-buffers to screen-space shading", "Single frame; photographic goal"],
    ["Neural Supersampling, 2020 [4]", "Warps past frames by engine motion + CNN", "Engine look only, no style"],
    ["Photorealism Enhancement, 2023 [5]", "G-buffer-conditioned, adversarial enhancement", "Real-photo domain, not artistic"],
    ["pix2pix, 2017 [6]", "Conditional GAN with a U-Net generator", "Needs paired data; per frame"],
    ["vid2vid / WC-vid2vid, 2018–20 [7], [8]", "Flow warping; 3D-guided consistency", "Estimated flow; heavy model"],
    ["Lai et al., 2018 [9]", "Post-hoc temporal-consistency network", "Estimated flow; extra pass"],
    ["GameNGen / Genie, 2024–25 [10], [11]", "Generative game engines", "No authoritative game state"],
    ["ControlNet, 2023 [12]", "Depth-controlled diffusion", "Slow; no temporal consistency"],
  ];
  s.addTable([head, ...rows], { x: 0.55, y: 1.45, w: 12.2, colW: [4.0, 4.5, 3.7], fontSize: 15, color: INK, fontFace: "Times New Roman",
    border: { type: "solid", pt: 0.75, color: "BFBFBF" }, rowH: 0.56, valign: "middle", margin: [0.03, 0.08, 0.03, 0.08] });
}

// ================================================================== 6. Architecture
{
  const s = slide("System Architecture", "Offline: record, teacher, gates, train, export. Runtime: one loop per game frame inside Godot.", "Method");
  img(s, LAB("architecture.png"), 0.55, 1.4, 7.4, 1279 / 957, "System architecture: offline data and supervision, and the runtime loop");
  leadBullets(s, [
    ["Offline", "record play sessions, make teacher targets, check them with 4 alignment gates, train the U-Net, export to ONNX and TensorRT."],
    ["Runtime", "every frame Godot captures 4 passes, C++ computes motion and the new-entity mask, TensorRT renders, Godot displays."],
    ["Fallback", "any renderer error shows the native frame, so the game never waits."],
  ], { x: 8.3, y: 1.5, w: 4.45, h: 5.3, fontSize: 18 });
}

// ================================================================== 7-9. Methodology: the pipeline from start to end
{
  const s = slide("Proposed Methodology (1/3)", "Stages 1-3: from the game to a validated dataset.");
  pipeline(s, 0, 2);
  leadBullets(s, [
    ["1. Godot game", "an arena collector (player, coins, spawned enemies) running physics at 60 Hz; the engine owns all game state."],
    ["2. Structural export", "override-shader passes render depth, normals, albedo and an entity-ID map; motion is computed analytically."],
    ["3. Recording and validation", "scripted play advances one physics step per frame; a validator rejects bad sessions. 24 sessions × 990 frames = 23,760 frames, split 18 / 3 / 3 by session."],
  ], { x: 0.55, y: 2.0, w: 5.9, h: 4.8, fontSize: 18 });
  img(s, LAB("inputs.png"), 6.7, 2.1, 6.05, 1279 / 438, "One test frame: native render, input channels, entity IDs, motion, confidence mask, teacher target");
  caption(s, "Exported for one test frame: native render, depth, normals, albedo, entity IDs, motion, confidence mask, oil target.", 6.7, 4.25, 6.05, 0.7);
}
{
  const s = slide("Proposed Methodology (2/3)", "Stages 4-6: targets, gates and the network.");
  pipeline(s, 3, 5);
  leadBullets(s, [
    ["4. Teacher targets", "Stable Diffusion 1.5 + depth ControlNet [12], [13] restyles each frame offline, guided by the engine's depth (oil 0.50, watercolour 0.55)."],
    ["5. Alignment gates", "boundary F ≥ 0.60, spurious edges ≤ 0.35, silhouettes ≥ 0.60, stylisation ≥ 0.15; 11 of 23,760 frames excluded."],
    ["6. Training", "StyleTemporalUNet [14], [15] (8.39 M parameters), loss L1 + 0.5 LPIPS [16] + 0.5 masked temporal; AdamW, 60 epochs, 29.2 h."],
  ], { x: 0.55, y: 2.0, w: 5.9, h: 4.8, fontSize: 18 });
  img(s, LAB("network.png"), 6.7, 2.0, 6.05, 1279 / 847, "StyleTemporalUNet architecture diagram");
}
{
  const s = slide("Proposed Methodology (3/3)", "Stages 7-8: from the trained model to the live game.");
  pipeline(s, 6, 7);
  leadBullets(s, [
    ["7. Deploy", "one live step of the model, including the confidence mask, is exported to ONNX and compiled to a TensorRT FP16 engine per style [18]."],
    ["8. Live runtime", "a C++ GDExtension inside Godot [19] runs the engine every frame with the previous frame's features; any error shows the native frame."],
    ["Check", "the live frame matches the evaluated model to within one colour level in 255."],
  ], { x: 0.55, y: 2.0, w: 5.9, h: 4.8, fontSize: 18 });
  img(s, LAB("live.png"), 6.7, 2.2, 6.05, 1578 / 454, "Live frame rate of the whole loop and TensorRT latency");
  caption(s, "Live frame rate in the game (target 30 fps) and TensorRT time per step (budget 20 ms).", 6.7, 4.0, 6.05, 0.6);
}

// ================================================================== 10. Algorithm & techniques
{
  const s = slide("Algorithms and Techniques Used", "The per-frame algorithm, and the techniques it relies on.");
  s.addText("Per-frame runtime algorithm", { isTextBox: true, x: 0.55, y: 1.45, w: 6.6, h: 0.5, fontSize: 22, bold: true, color: MAROON, margin: 0 });
  const steps = [
    "Godot steps physics and captures depth, normals, albedo and entity-ID passes.",
    "C++ computes motion vectors and the no-history mask for new entities.",
    "Previous features are warped by engine motion and masked.",
    "TensorRT FP16 engine renders the frame in the selected style.",
    "Display the frame and keep its features for the next frame.",
    "On any error, show the native frame; gameplay never waits.",
  ];
  s.addText(steps.map((t, i) => ({ text: `${i + 1}.  ${t}`, options: { breakLine: i < steps.length - 1 } })), // numbered by hand
    { isTextBox: true, x: 0.55, y: 2.05, w: 6.6, h: 4.7, fontSize: 18, color: INK, valign: "top", paraSpaceAfter: 10, margin: 0.05 });
  s.addText("Techniques", { isTextBox: true, x: 7.5, y: 1.45, w: 5.25, h: 0.5, fontSize: 22, bold: true, color: MAROON, margin: 0 });
  bullets(s, ["U-Net convolutional network", "FiLM style conditioning", "Engine motion-vector warping", "Confidence (disocclusion) masking",
    "Stable Diffusion 1.5 + ControlNet", "LPIPS perceptual loss", "AdamW with cosine learning rate", "bfloat16 mixed precision",
    "ONNX export, TensorRT FP16", "C++ GDExtension in Godot 4"], { x: 7.5, y: 2.05, w: 5.25, h: 4.7, fontSize: 18, paraSpaceAfter: 4 });
}

// ================================================================== 11-12. Results
{
  const s = slide("Results: Image Quality", "Held-out game test sessions, 2,968 frames per style.", "Results");
  const stats = [["35.33 dB", "PSNR, oil"], ["34.79 dB", "PSNR, watercolour"], ["0.54× / 0.48×", "flicker vs. its own teacher"], ["7.3–7.8 dB", "better than showing the native frame"]];
  stats.forEach(([v, l], i) => {
    const x = 0.55 + i * 3.08;
    s.addText([{ text: v, options: { fontSize: 32, bold: true, color: MAROON, breakLine: true } }, { text: l, options: { fontSize: 16, color: INK } }],
      { isTextBox: true, x, y: 1.5, w: 2.9, h: 1.3, align: "left", valign: "top", margin: 0, objectName: "stat " + (i + 1) });
  });
  const head = ["Model, style", "PSNR (dB)", "SSIM", "LPIPS", "Sharpness", "Colour travel", "Flicker vs teacher"].map((t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: MAROON } } }));
  const rows = [
    ["Two-style model, oil", "35.33", "0.9807", "0.0225", "0.946", "0.995", "0.54×"],
    ["Two-style model, watercolour", "34.79", "0.9786", "0.0243", "0.937", "0.987", "0.48×"],
    ["Oil-only model (reference)", "35.43", "0.9814", "0.0216", "0.946", "0.980", "0.54×"],
    ["Native frame (do nothing)", "28.06 / 26.96", "–", "–", "–", "–", "–"],
  ];
  s.addTable([head, ...rows], { x: 0.55, y: 3.05, w: 12.2, colW: [3.6, 1.6, 1.2, 1.2, 1.4, 1.6, 1.6], fontSize: 16, color: INK, fontFace: "Times New Roman",
    border: { type: "solid", pt: 0.75, color: "BFBFBF" }, rowH: 0.5, valign: "middle", align: "center" });
  caption(s, "Gates: sharpness ≥ 0.90 and colour travel ≥ 0.80; both styles pass.", 0.55, 5.75, 12.2, 0.4);
}
{
  const s = slide("Results: Ablation and Live Speed", "The plain U-Net beats the transformer; the live loop runs well above 30 fps.");
  const chartBase = { catAxisLabelColor: INK, valAxisLabelColor: INK, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
    dataLabelFontFace: "+mn-lt", titleFontFace: "+mn-lt", legendFontFace: "+mn-lt", titleColor: INK, dataLabelColor: INK,
    catAxisLabelFontSize: 14, valAxisLabelFontSize: 12, dataLabelFontSize: 13, titleFontSize: 18, showTitle: true,
    valGridLine: { color: "E0E0E0", size: 0.5 }, catGridLine: { style: "none" }, showValue: true, dataLabelPosition: "outEnd" };
  s.addChart(pres.charts.BAR, [{ name: "Sharpness", labels: ["A U-Net", "B U-Net + temporal", "C StructFormer", "D StructFormer + temporal", "E + entity", "N raw-ID control"],
    values: [0.912, 0.897, 0.853, 0.858, 0.884, 0.880] }],
  { ...chartBase, x: 0.55, y: 1.45, w: 6.0, h: 5.2, barDir: "bar", title: "Sharpness on the ablation test set (gate 0.90)", chartColors: [MAROON],
    valAxisMinVal: 0.8, valAxisMaxVal: 0.94, valAxisMajorUnit: 0.02, dataLabelFormatCode: "0.000", showLegend: false, catAxisOrientation: "maxMin" });
  s.addChart(pres.charts.BAR, [
    { name: "Oil", labels: ["No history", "History, FP32", "History, FP16 (shipped)"], values: [50.2, 41.8, 44.6] },
    { name: "Watercolour", labels: ["No history", "History, FP32", "History, FP16 (shipped)"], values: [45.8, 41.0, 44.7] }],
  { ...chartBase, x: 6.85, y: 1.45, w: 5.9, h: 5.2, barDir: "col", barGrouping: "clustered", title: "Live frames per second (target 30)",
    chartColors: [MAROON, "8C8C8C"], valAxisMinVal: 0, valAxisMaxVal: 60, valAxisMajorUnit: 10, dataLabelFormatCode: "0.0", showLegend: true, legendPos: "b", legendFontSize: 14 });
}

// ================================================================== 13-14. Output
{
  const s = slide("Output: StructRender Live", "Real output of the trained model, live in the game and on a sealed test frame.", "Output");
  img(s, PAPER("teaser.png"), 0.55, 1.4, 12.2, 2166 / 449, "Native frame and live StructRender oil and watercolour");
  img(s, LAB("qualitative.png"), 2.4, 4.0, 6.4, 1279 / 546, "Native input, teacher target and model output for both styles");
  caption(s, "Top: one live game frame, native vs. StructRender oil and watercolour (TensorRT FP16, 44.6 fps). Bottom: sealed test frame, native input, teacher target and model output.",
    9.05, 4.1, 3.7, 2.6);
}
{
  const s = slide("Output: Web Game Demo", "A separate browser web-swinging game (structrender.vercel.app) with 16 hand-made shader themes. These images are not output of the StructRender model.");
  img(s, LAB("collage_street.jpg"), 0.55, 1.4, 6.0, 1944 / 1382, "Street scene: original look and 16 visual themes in the web game");
  img(s, LAB("collage_swing.jpg"), 6.75, 1.4, 6.0, 1944 / 1382, "Swinging scene: original look and 16 visual themes in the web game");
  caption(s, "Separate Three.js web game (structrender.vercel.app): original look and 16 hand-made shader themes. Not output of the StructRender model.",
    0.55, 5.8, 12.2, 0.6);
}

// ================================================================== 15-16. Future scope, conclusion
{
  const s = slide("Future Scope", "Four directions that follow directly from the limitations.", "Conclusion");
  leadBullets(s, [
    ["Consistent teacher", "video-consistent diffusion or authored textures, which limit both stability and EPAM today."],
    ["Re-test EPAM", "scenes with many entities, occlusions and recycled IDs, as a new pre-registered experiment."],
    ["Higher resolution", "super-resolution from 512 × 288 targets, or new styles designed at 768 × 432."],
    ["Broader evaluation", "more games and textured scenes, human play, repeated seeds, INT8/FP8 inference."],
  ], { x: 0.55, y: 1.5, w: 12.2, h: 5.3, fontSize: 22, paraSpaceAfter: 14 });
}
{
  const s = slide("Conclusion", "What was delivered and what was learned.");
  bullets(s, [
    "A learned painted style was added to a playable game, with the engine keeping all game state.",
    "Gated diffusion supervision and a temporal two-style U-Net reach 35.33 / 34.79 dB and sharpness 0.946 / 0.937.",
    "The renderer flickers about half as much as its own teacher (0.54× and 0.48×).",
    "It runs live at 44.6 fps (4.53 ms per TensorRT FP16 step), with 0 fallbacks in 3,000 frames.",
    "The live frame matches the evaluated model to within one colour level.",
    "Honest negatives: the transformer [17] did not beat the U-Net, and EPAM did not improve output on this data.",
  ], { x: 0.55, y: 1.5, w: 12.2, h: 5.3, fontSize: 21, paraSpaceAfter: 10 });
}

// ================================================================== 17-18. References (IEEE)
const REFS = [
  'A. Tewari et al., "State of the art on neural rendering," Comput. Graph. Forum, vol. 39, no. 2, pp. 701–727, 2020.',
  'T. Saito and T. Takahashi, "Comprehensible rendering of 3-D shapes," ACM SIGGRAPH Comput. Graph., vol. 24, no. 4, pp. 197–206, 1990.',
  'O. Nalbach, E. Arabadzhiyska, D. Mehta, H.-P. Seidel, and T. Ritschel, "Deep shading: Convolutional neural networks for screen space shading," Comput. Graph. Forum, vol. 36, no. 4, pp. 65–78, 2017.',
  'L. Xiao, S. Nouri, M. Chapman, A. Fix, D. Lanman, and A. Kaplanyan, "Neural supersampling for real-time rendering," ACM Trans. Graph., vol. 39, no. 4, 2020.',
  'S. R. Richter, H. A. Alhaija, and V. Koltun, "Enhancing photorealism enhancement," IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 2, pp. 1700–1715, 2023.',
  'P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, "Image-to-image translation with conditional adversarial networks," in Proc. IEEE CVPR, 2017, pp. 5967–5976.',
  'T.-C. Wang et al., "Video-to-video synthesis," in Advances in Neural Information Processing Systems (NeurIPS), 2018.',
  'A. Mallya, T.-C. Wang, K. Sapra, and M.-Y. Liu, "World-consistent video-to-video synthesis," in Proc. ECCV, 2020, pp. 359–378.',
  'W.-S. Lai et al., "Learning blind video temporal consistency," in Proc. ECCV, 2018, pp. 179–195.',
  'D. Valevski, Y. Leviathan, M. Arar, and S. Fruchter, "Diffusion models are real-time game engines," in Proc. ICLR, 2025.',
  'J. Bruce et al., "Genie: Generative interactive environments," arXiv:2402.15391, 2024.',
  'L. Zhang, A. Rao, and M. Agrawala, "Adding conditional control to text-to-image diffusion models," in Proc. IEEE/CVF ICCV, 2023, pp. 3813–3824.',
  'R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, "High-resolution image synthesis with latent diffusion models," in Proc. IEEE/CVF CVPR, 2022, pp. 10674–10685.',
  'O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," in Proc. MICCAI, 2015, pp. 234–241.',
  'E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, "FiLM: Visual reasoning with a general conditioning layer," in Proc. AAAI, vol. 32, no. 1, 2018.',
  'R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, "The unreasonable effectiveness of deep features as a perceptual metric," in Proc. IEEE/CVF CVPR, 2018, pp. 586–595.',
  'Z. Liu et al., "Swin Transformer: Hierarchical vision transformer using shifted windows," in Proc. IEEE/CVF ICCV, 2021, pp. 9992–10002.',
  'NVIDIA Corporation, "NVIDIA TensorRT documentation." [Online]. Available: https://docs.nvidia.com/deeplearning/tensorrt/latest/',
  'Godot Engine contributors, "Godot Engine documentation." [Online]. Available: https://docs.godotengine.org/en/stable/',
];
[[0, 10], [10, REFS.length]].forEach(([a, b], k) => {
  const s = slide(`References (${k + 1}/2)`, "", k === 0 ? "References" : undefined);
  s.addText(REFS.slice(a, b).map((r, i) => ({ text: `[${a + i + 1}]  ${r}`, options: { breakLine: i < b - a - 1 } })),
    { isTextBox: true, x: 0.55, y: 1.45, w: 12.2, h: 5.35, fontSize: 15, color: INK, valign: "top", paraSpaceAfter: 7, margin: 0.03 });
});

// ================================================================== 19. Thank you
{
  const s = pres.addSlide({ masterName: "TITLE", sectionTitle: section });
  s.addText("Thank You", { isTextBox: true, x: 0.6, y: 2.6, w: 12.13, h: 1.4, fontSize: 60, bold: true, color: MAROON, align: "center", valign: "middle", margin: 0 });
  s.addText("Questions?", { isTextBox: true, x: 0.6, y: 4.0, w: 12.13, h: 0.8, fontSize: 28, color: INK, align: "center", valign: "middle", margin: 0 });
}

pres.writeFile({ fileName: OUT }).then(() => applyTheme(OUT, THEME)).then(() => console.log("wrote", OUT));

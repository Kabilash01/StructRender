// Renders docs/paper/build/content.js twice: to an Overleaf-ready LaTeX project (IEEEtran conference
// class) and to an IEEE-style two-column Word document. One numbering pass serves both, so figure,
// table, equation and citation numbers agree between the two.
// Run: NODE_PATH=<dir with docx@9> node docs/paper/build/build_paper.js <overleaf_dir> <out.docx>
"use strict";
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, BorderStyle,
  AlignmentType, TabStopType, SectionType, LevelFormat, VerticalAlign,
} = require("docx");
const { meta, refs, blocks } = require("./content");

const ROOT = path.resolve(__dirname, "../../..");
const FIGDIR = path.join(ROOT, "docs/paper/figures");
const [OVERLEAF, DOCX_OUT] = process.argv.slice(2);
if (!OVERLEAF || !DOCX_OUT) throw new Error("usage: build_paper.js <overleaf_dir> <out.docx>");

// ------------------------------------------------------------------ numbering (shared)
const ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"];
const xref = {};
{
  let s = 0, sub = 0, f = 0, t = 0, e = 0;
  for (const b of blocks) {
    if (b.t === "h1") { b.num = ROMAN[s++]; sub = 0; if (b.label) xref[b.label] = { kind: "sec", n: b.num }; }
    else if (b.t === "h2") b.num = String.fromCharCode(65 + sub++);
    else if (b.t === "fig") xref[b.label] = { kind: "fig", n: (b.num = ++f) };
    else if (b.t === "table") xref[b.label] = { kind: "tab", n: (b.num = ROMAN[t++]) };
    else if (b.t === "eq") xref[b.label] = { kind: "eq", n: (b.num = ++e) };
  }
}

// ------------------------------------------------------------------ inline markup
const RE = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|_\{[^}]*\}|\^\{[^}]*\}|\[@[^\]]+\]|\{\{[a-z]+:[^}]+\}\})/g;
function parse(text) {
  const out = [];
  let last = 0, m;
  const re = new RegExp(RE.source, "g");
  while ((m = re.exec(text))) {
    if (m.index > last) out.push({ k: "text", v: text.slice(last, m.index) });
    const t = m[0];
    if (t.startsWith("**")) out.push({ k: "bold", c: parse(t.slice(2, -2)) });
    else if (t.startsWith("`")) out.push({ k: "code", v: t.slice(1, -1) });
    else if (t.startsWith("[@")) out.push({ k: "cite", keys: t.slice(1, -1).split(",").map((s) => s.trim().replace(/^@/, "")) });
    else if (t.startsWith("{{")) out.push({ k: "ref", label: t.slice(2, -2) });
    else if (t.startsWith("*")) out.push({ k: "italic", c: parse(t.slice(1, -1)) });
    else if (t.startsWith("_{")) out.push({ k: "sub", c: parse(t.slice(2, -1)) });
    else out.push({ k: "sup", c: parse(t.slice(2, -1)) });
    last = m.index + t.length;
  }
  if (last < text.length) out.push({ k: "text", v: text.slice(last) });
  return out;
}

// citation order = first appearance in document order (IEEE numbering)
const order = [];
function collectToks(toks) {
  for (const tok of toks) {
    if (tok.k === "cite") tok.keys.forEach((k) => {
      if (!refs[k]) throw new Error("unknown reference " + k);
      if (!order.includes(k)) order.push(k);
    });
    if (tok.k === "ref" && !xref[tok.label]) throw new Error("unknown cross-reference " + tok.label);
    if (tok.c) collectToks(tok.c);
  }
}
const collect = (text) => collectToks(parse(text));
for (const b of blocks) {
  const texts = b.t === "list" ? b.items : b.t === "table" ? [b.caption, ...b.rows.flat()] : b.t === "eq" ? [b.text]
    : [b.text || b.caption || ""];
  texts.forEach(collect);
}
const unused = Object.keys(refs).filter((k) => !order.includes(k));
if (unused.length) throw new Error("uncited references: " + unused.join(", "));
const citeNum = (k) => order.indexOf(k) + 1;
function citeLabel(keys) { // IEEE / cite.sty: sorted, runs of 3+ compressed to [a]–[c]
  const n = [...new Set(keys.map(citeNum))].sort((a, b) => a - b), parts = [];
  for (let i = 0; i < n.length;) {
    let j = i;
    while (j + 1 < n.length && n[j + 1] === n[j] + 1) j++;
    if (j - i >= 2) parts.push(`[${n[i]}]–[${n[j]}]`); else for (let x = i; x <= j; x++) parts.push(`[${n[x]}]`);
    i = j + 1;
  }
  return parts.join(", ");
}
const refText = (label) => {
  const r = xref[label];
  return { fig: `Fig. ${r.n}`, tab: `Table ${r.n}`, eq: `(${r.n})`, sec: `Section ${r.n}` }[r.kind];
};

// =================================================================== LaTeX
const UNI = { // pdfLaTeX has no glyphs for these in text mode
  "−": "\\ensuremath{-}", "×": "\\ensuremath{\\times}", "≥": "\\ensuremath{\\geq}", "≤": "\\ensuremath{\\leq}",
  "→": "\\ensuremath{\\rightarrow}", "⊙": "\\ensuremath{\\odot}", "‖": "\\ensuremath{\\|}", "Σ": "\\ensuremath{\\Sigma}",
  "α": "\\ensuremath{\\alpha}", "β": "\\ensuremath{\\beta}", "γ": "\\ensuremath{\\gamma}", "κ": "\\ensuremath{\\kappa}",
  "λ": "\\ensuremath{\\lambda}", "ρ": "\\ensuremath{\\rho}", "σ": "\\ensuremath{\\sigma}", "Δ": "\\ensuremath{\\Delta}",
  "∇": "\\ensuremath{\\nabla}", "∞": "\\ensuremath{\\infty}", "∈": "\\ensuremath{\\in}", "√": "\\ensuremath{\\surd}",
  "°": "\\ensuremath{^\\circ}", "·": "\\ensuremath{\\cdot}", "′": "\\ensuremath{'}", "𝒲": "\\ensuremath{\\mathcal{W}}",
  "ℒ": "\\ensuremath{\\mathcal{L}}",
};
const LATIN = /[\u00c0-\u00ff–—’‘“”]/; // handled by inputenc/T1
function esc(s) {
  return s.replace(/[\\&%$#_{}~^]/g, (c) => ({ "\\": "\\textbackslash{}", "&": "\\&", "%": "\\%", "$": "\\$", "#": "\\#",
    "_": "\\_", "{": "\\{", "}": "\\}", "~": "\\textasciitilde{}", "^": "\\textasciicircum{}" })[c])
    .replace(/"([^"]*)"/g, "``$1''").replace(/\b(vs|al)\. /g, "$1.\\ ");
}
function tex(text) {
  return parse(text).map((t) => {
    switch (t.k) {
      case "text": return esc(t.v);
      case "bold": return `\\textbf{${t.c.map(texTok).join("")}}`;
      case "italic": return `\\textit{${t.c.map(texTok).join("")}}`;
      case "code": return `\\texttt{${esc(t.v)}}`;
      case "sub": return `\\textsubscript{${t.c.map(texTok).join("")}}`;
      case "sup": return `\\textsuperscript{${t.c.map(texTok).join("")}}`;
      case "cite": return `\\cite{${t.keys.join(",")}}`;
      case "ref": {
        const r = xref[t.label];
        return { fig: `Fig.~\\ref{${t.label}}`, tab: `Table~\\ref{${t.label}}`, eq: `(\\ref{${t.label}})`,
          sec: `Section~\\ref{${t.label}}` }[r.kind];
      }
    }
    return "";
  }).join("");
}
function texTok(t) { return tex(serialize(t)); }
function serialize(t) { // back to markup, so nested tokens reuse tex()
  switch (t.k) {
    case "text": return t.v;
    case "bold": return "**" + t.c.map(serialize).join("") + "**";
    case "italic": return "*" + t.c.map(serialize).join("") + "*";
    case "code": return "`" + t.v + "`";
    case "sub": return "_{" + t.c.map(serialize).join("") + "}";
    case "sup": return "^{" + t.c.map(serialize).join("") + "}";
    case "cite": return "[" + t.keys.map((k) => "@" + k).join(",") + "]";
    case "ref": return "{{" + t.label + "}}";
  }
  return "";
}
function refTex(s) { // URLs into \url{}, the rest escaped
  return s.split(/(https?:\/\/\S+)/).map((part, i) => i % 2 ? `\\url{${part}}` : tex(part)).join("");
}
function figFile(name) {
  for (const ext of [".pdf", ".png"]) if (fs.existsSync(path.join(FIGDIR, name + ext))) return name + ext;
  throw new Error("missing figure " + name);
}

function buildLatex() {
  const L = [];
  const used = new Set();
  L.push("% StructRender conference paper draft (IEEEtran). Generated by docs/paper/build/build_paper.js from content.js.");
  L.push("% Compile with pdfLaTeX (Overleaf default). Placeholders marked [DETAILS REQUIRED] must be filled in.");
  L.push("\\documentclass[conference]{IEEEtran}");
  L.push("\\IEEEoverridecommandlockouts");
  L.push("\\usepackage[T1]{fontenc}");
  L.push("\\usepackage{cite}");
  L.push("\\usepackage{amsmath,amssymb}");
  L.push("\\usepackage{graphicx}");
  L.push("\\usepackage{booktabs,tabularx,array}");
  L.push("\\usepackage{textcomp}");
  L.push("\\usepackage{url}");
  const uniLines = [];
  L.push("%UNICODE%");
  L.push("\\begin{document}");
  L.push(`\\title{${tex(meta.title)}}`);
  L.push("\\author{" + meta.authors.map((a) => `\\IEEEauthorblockN{${tex(a.name)}}\n\\IEEEauthorblockA{${a.affil.flatMap((l) => l.split("\n")).map(tex).join("\\\\\\relax\n")}}`).join("\n\\and\n") + "}");
  L.push("\\maketitle");
  L.push(`\\begin{abstract}\n${tex(meta.abstract)}\n\\end{abstract}`);
  L.push(`\\begin{IEEEkeywords}\n${tex(meta.keywords)}\n\\end{IEEEkeywords}`);
  for (const b of blocks) {
    if (b.t === "h1") L.push(`\n\\section{${tex(b.text)}}${b.label ? `\\label{${b.label}}` : ""}`);
    else if (b.t === "h2") L.push(`\n\\subsection{${tex(b.text)}}`);
    else if (b.t === "p") L.push("\n" + tex(b.text));
    else if (b.t === "list") L.push("\\begin{itemize}\n" + b.items.map((i) => "\\item " + tex(i)).join("\n") + "\n\\end{itemize}");
    else if (b.t === "eq") L.push(`\\begin{equation}\\label{${b.label}}\n${b.tex}\n\\end{equation}`);
    else if (b.t === "fig") {
      const env = b.wide ? "figure*" : "figure";
      const w = b.wide ? `${b.width || 1}\\textwidth` : "\\columnwidth";
      L.push(`\\begin{${env}}[!t]\n\\centering\n\\includegraphics[width=${w}]{figures/${figFile(b.file)}}\n\\caption{${tex(b.caption)}}\\label{${b.label}}\n\\end{${env}}`);
      used.add(figFile(b.file));
    } else if (b.t === "table") {
      const env = b.wide ? "table*" : "table";
      const hasX = b.cols.some((c) => c.a === "X");
      const spec = b.cols.map((c) => c.a).join("");
      const open = hasX ? `\\begin{tabularx}{${b.wide ? "\\textwidth" : "\\columnwidth"}}{${spec}}` : `\\begin{tabular}{${spec}}`;
      const close = hasX ? "\\end{tabularx}" : "\\end{tabular}";
      const row = (cells) => cells.map(tex).join(" & ") + " \\\\";
      L.push([`\\begin{${env}}[!t]`, `\\caption{${tex(b.caption)}}\\label{${b.label}}`, "\\centering",
        `\\${b.size || "footnotesize"}`, "\\setlength{\\tabcolsep}{" + (b.size === "scriptsize" ? "2.5pt" : "4pt") + "}",
        open, "\\toprule", row(b.cols.map((c) => c.h)), "\\midrule", ...b.rows.map(row), "\\bottomrule", close,
        `\\end{${env}}`].join("\n"));
    }
  }
  L.push("\n\\begin{thebibliography}{" + order.length + "}");
  order.forEach((k) => L.push(`\\bibitem{${k}} ${refTex(refs[k])}`));
  L.push("\\end{thebibliography}");
  L.push("\\end{document}");
  let src = L.join("\n");
  // declare exactly the Unicode characters the text uses; fail on any the table does not cover
  const chars = [...new Set([...src].filter((c) => c.codePointAt(0) > 127))];
  const missing = chars.filter((c) => !UNI[c] && !LATIN.test(c));
  if (missing.length) throw new Error("no LaTeX mapping for: " + missing.join(" "));
  chars.filter((c) => UNI[c]).forEach((c) =>
    uniLines.push(`\\DeclareUnicodeCharacter{${c.codePointAt(0).toString(16).toUpperCase().padStart(4, "0")}}{${UNI[c]}}`));
  src = src.replace("%UNICODE%", uniLines.join("\n"));
  fs.mkdirSync(path.join(OVERLEAF, "figures"), { recursive: true });
  fs.writeFileSync(path.join(OVERLEAF, "main.tex"), src + "\n");
  for (const f of used) fs.copyFileSync(path.join(FIGDIR, f), path.join(OVERLEAF, "figures", f));
  console.log("wrote", path.join(OVERLEAF, "main.tex"), "figures:", used.size, "references:", order.length);
}

// =================================================================== Word (IEEE-style layout)
const FONT = "Times New Roman";
const PAGE = { size: { width: 12240, height: 15840 }, margin: { top: 1080, bottom: 1440, left: 900, right: 900 } };
const FULL = 12240 - 1800, GAP = 360, COL = (FULL - GAP) / 2; // DXA: 7.25 in text, 3.5 in columns
const DOCX_CHARS = { "𝒲": "W", "ℒ": "L" }; // script letters missing from Times New Roman
const dch = (s) => s.replace(/[𝒲ℒ]/gu, (c) => DOCX_CHARS[c]);

function runs(text, base = {}) {
  const out = [];
  const walk = (toks, b) => toks.forEach((t) => {
    switch (t.k) {
      case "text": out.push(new TextRun({ text: dch(t.v), ...b })); break;
      case "bold": walk(t.c, { ...b, bold: true }); break;
      case "italic": walk(t.c, { ...b, italics: true }); break;
      case "code": out.push(new TextRun({ text: t.v, ...b, font: "Courier New", size: (b.size || 20) - 2 })); break;
      case "sub": walk(t.c, { ...b, subScript: true }); break;
      case "sup": walk(t.c, { ...b, superScript: true }); break;
      case "cite": out.push(new TextRun({ text: citeLabel(t.keys), ...b })); break;
      case "ref": out.push(new TextRun({ text: refText(t.label), ...b })); break;
    }
  });
  walk(parse(text), base);
  return out;
}
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

function buildDocx() {
  const sections = [];
  let cur = { cols: 1, children: [] };
  const setCols = (n) => {
    if (cur.cols === n) return;
    if (cur.children.length) sections.push(cur);
    cur = { cols: n, children: [] };
  };
  const add = (...xs) => cur.children.push(...xs);
  const para = (text, o = {}, base = {}) => new Paragraph({ children: runs(text, base), ...o });

  // title block (one column)
  add(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
    children: [new TextRun({ text: meta.title, size: 48 })] }));
  const noLine = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const aw = Math.floor(FULL / meta.authors.length);
  add(new Table({ width: { size: FULL, type: WidthType.DXA }, columnWidths: meta.authors.map(() => aw), rows: [
    new TableRow({ children: meta.authors.map((a) => new TableCell({ width: { size: aw, type: WidthType.DXA },
      borders: { top: noLine, bottom: noLine, left: noLine, right: noLine },
      children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: a.name, size: 22 })] }),
        ...a.affil.flatMap((l) => l.split("\n")).map((l) => new Paragraph({ alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: l, size: 20, italics: !l.includes("@") })] }))] })) }),
  ] }));
  add(new Paragraph({ spacing: { after: 240 }, children: [] }));
  setCols(2);
  add(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 }, children: [
    new TextRun({ text: "Abstract", bold: true, italics: true, size: 18 }), new TextRun({ text: "—", bold: true, size: 18 }),
    ...runs(meta.abstract, { bold: true, size: 18 })] }));
  add(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 160 }, children: [
    new TextRun({ text: "Index Terms", bold: true, italics: true, size: 18 }), new TextRun({ text: "—", bold: true, size: 18 }),
    ...runs(meta.keywords, { bold: true, size: 18 })] }));

  let firstPara = false;
  for (const b of blocks) {
    if (b.t === "h1") {
      add(new Paragraph({ style: "H1", children: [new TextRun({ text: `${b.num}. ${b.text}`, smallCaps: true })] }));
      firstPara = true;
    } else if (b.t === "h2") {
      add(new Paragraph({ style: "H2", children: [new TextRun({ text: `${b.num}. ${b.text}`, italics: true })] }));
      firstPara = true;
    } else if (b.t === "p") {
      add(para(b.text, { style: "Body", indent: firstPara ? { firstLine: 0 } : undefined }));
      firstPara = false;
    } else if (b.t === "list") {
      b.items.forEach((i) => add(para(i, { style: "Body", numbering: { reference: "bul", level: 0 } })));
    } else if (b.t === "eq") {
      const w = cur.cols === 2 ? COL : FULL;
      const lines = b.text.split("\n"); // a long equation is broken by hand; the number goes on the last line
      add(new Paragraph({ style: "Body", indent: { firstLine: 0 }, spacing: { before: 80, after: 80 },
        tabStops: [{ type: TabStopType.CENTER, position: w / 2 }, { type: TabStopType.RIGHT, position: w }],
        children: lines.flatMap((l, i) => [new TextRun({ text: "\t", break: i ? 1 : 0 }), ...runs(l),
          ...(i === lines.length - 1 ? [new TextRun(`\t(${b.num})`)] : [])]) }));
      firstPara = true;
    } else if (b.t === "fig") {
      if (b.wide) setCols(1);
      const file = path.join(FIGDIR, b.file + ".png");
      const { w, h } = pngSize(file);
      const inches = b.wide ? 7.16 * (b.width || 1) : 3.45;
      add(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
        children: [new ImageRun({ type: "png", data: fs.readFileSync(file),
          transformation: { width: Math.round(inches * 96), height: Math.round((inches * 96 * h) / w) },
          altText: { title: `Fig. ${b.num}`, description: b.caption.replace(/[*_{}]/g, ""), name: b.file } })] }));
      add(new Paragraph({ style: "FigCap", children: [new TextRun({ text: `Fig. ${b.num}. ` }), ...runs(b.caption)] }));
      if (b.wide) setCols(2);
    } else if (b.t === "table") {
      if (b.wide) setCols(1);
      const W = b.wide ? FULL : COL, sz = b.size === "scriptsize" ? 14 : 16;
      const ws = b.cols.map((c) => Math.round((W * c.w) / 100));
      const al = { l: AlignmentType.LEFT, X: AlignmentType.LEFT, c: AlignmentType.CENTER, r: AlignmentType.RIGHT };
      const line = (s) => ({ style: BorderStyle.SINGLE, size: s, color: "000000" });
      const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
      const n = b.rows.length;
      const cell = (text, i, ri) => new TableCell({ width: { size: ws[i], type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
        margins: { top: 20, bottom: 20, left: 60, right: 60 },
        borders: { left: none, right: none, top: ri === -1 ? line(8) : none,
          bottom: ri === -1 ? line(4) : ri === n - 1 ? line(8) : none },
        children: [new Paragraph({ alignment: al[b.cols[i].a], keepNext: ri < n - 1, children: runs(text, { size: sz }) })] });
      add(new Paragraph({ style: "TabCap", keepNext: true, spacing: { before: 160, after: 0 }, children: [new TextRun(`TABLE ${b.num}`)] }));
      add(new Paragraph({ style: "TabCap", keepNext: true, spacing: { after: 60 }, children: runs(b.caption, { smallCaps: true }) }));
      add(new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: ws, alignment: AlignmentType.CENTER, rows: [
        new TableRow({ tableHeader: true, cantSplit: true, children: b.cols.map((c, i) => cell(c.h, i, -1)) }),
        ...b.rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, i, ri)) })),
      ] }));
      add(new Paragraph({ spacing: { after: 120 }, children: [] }));
      if (b.wide) setCols(2);
    }
  }
  add(new Paragraph({ style: "H1", children: [new TextRun({ text: "References", smallCaps: true })] }));
  order.forEach((k, i) => add(new Paragraph({ style: "Ref", tabStops: [{ type: TabStopType.LEFT, position: 400 }],
    children: [new TextRun(`[${i + 1}]\t`), ...runs(refs[k], { size: 16 })] })));
  sections.push(cur);

  const doc = new Document({
    creator: "StructRender", title: meta.title, description: "Conference paper draft",
    styles: {
      default: { document: { run: { font: FONT, size: 20 } } },
      paragraphStyles: [
        { id: "Body", name: "Body", basedOn: "Normal", run: { font: FONT, size: 20 },
          paragraph: { alignment: AlignmentType.JUSTIFIED, indent: { firstLine: 202 }, spacing: { before: 0, after: 0, line: 240 } } },
        { id: "H1", name: "H1", basedOn: "Normal", next: "Body", run: { font: FONT, size: 20 },
          paragraph: { alignment: AlignmentType.CENTER, keepNext: true, outlineLevel: 0, spacing: { before: 200, after: 100 } } },
        { id: "H2", name: "H2", basedOn: "Normal", next: "Body", run: { font: FONT, size: 20 },
          paragraph: { keepNext: true, outlineLevel: 1, spacing: { before: 120, after: 60 } } },
        { id: "FigCap", name: "FigCap", basedOn: "Normal", run: { font: FONT, size: 16 },
          paragraph: { alignment: AlignmentType.JUSTIFIED, spacing: { before: 0, after: 200 } } },
        { id: "TabCap", name: "TabCap", basedOn: "Normal", run: { font: FONT, size: 16 },
          paragraph: { alignment: AlignmentType.CENTER } },
        { id: "Ref", name: "Ref", basedOn: "Normal", run: { font: FONT, size: 16 },
          paragraph: { alignment: AlignmentType.JUSTIFIED, indent: { left: 400, hanging: 400 }, spacing: { after: 40 } } },
      ],
    },
    numbering: { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
      alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 200 } } } }] }] },
    sections: sections.map((s, i) => ({
      properties: { ...(i ? { type: SectionType.CONTINUOUS } : {}), page: PAGE,
        column: s.cols === 2 ? { count: 2, space: GAP, equalWidth: true } : { count: 1 } },
      children: s.children,
    })),
  });
  return Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(DOCX_OUT, buf); console.log("wrote", DOCX_OUT, "sections:", sections.length); });
}

buildLatex();
buildDocx();

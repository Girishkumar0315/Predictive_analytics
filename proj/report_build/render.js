// Renders content.json -> DOCX following the LPU report conventions (A4, TNR 12, 1.5 spacing).
const fs = require("fs");
const path = require("path");
const D = require("docx");
const { Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  AlignmentType, HeadingLevel, Footer, PageNumber, NumberFormat, TableOfContents, SequentialIdentifier, PageBreak,
  LevelFormat, VerticalAlign, TabStopType } = D;

const FONT = "Times New Roman";
const content = JSON.parse(fs.readFileSync(path.join(__dirname, "content.json"), "utf8"));
const FIG_DIR = path.join(__dirname, "..", "reports", "figures");
const LINE = 360; // 1.5 line spacing
const PAGE = { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } };
const TEXT_W = 11906 - 2880; // 9026 dxa

// ---- inline markup: **bold**, *italic*, `mono`, ^{sup}, _{sub}
function runs(text, base = {}) {
  const out = []; const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\^\{[^}]+\}|_\{[^}]+\})/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, ...base }));
    else if (t.startsWith("*")) out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: FONT, ...base }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), font: "Courier New", ...base, size: (base.size || 24) - 2 }));
    else if (t.startsWith("^")) out.push(new TextRun({ text: t.slice(2, -1), superScript: true, font: FONT, ...base }));
    else out.push(new TextRun({ text: t.slice(2, -1), subScript: true, font: FONT, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}
const sp = (before = 0, after = 120) => ({ before, after, line: LINE });

function pngSize(file) { const b = fs.readFileSync(file); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }

function caption(label, text) {
  return new Paragraph({ style: "Caption", alignment: AlignmentType.CENTER, keepNext: label === "Table",
    spacing: { before: label === "Table" ? 200 : 80, after: label === "Table" ? 80 : 240, line: 276 },
    children: [new TextRun({ text: `${label} `, bold: true, font: FONT, size: 22 }), new SequentialIdentifier(label),
      new TextRun({ text: `: ${text}`, font: FONT, size: 22 })] });
}

function block(b) {
  switch (b.t) {
    case "h1": return [new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: b.nobreak ? false : true, children: [new TextRun({ text: b.text, font: FONT })] })];
    case "h2": return [new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: b.text, font: FONT })] })];
    case "h3": return [new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({ text: b.text, font: FONT })] })];
    case "title": return [new Paragraph({ alignment: AlignmentType.CENTER, spacing: sp(0, 240), children: [new TextRun({ text: b.text, bold: true, font: FONT, size: 28 })] })];
    case "p": return [new Paragraph({ alignment: b.text.includes("`") ? AlignmentType.LEFT : AlignmentType.JUSTIFIED, spacing: sp(0, 120), children: runs(b.text) })];
    case "bullets": return b.items.map(it => new Paragraph({ numbering: { reference: "bul", level: 0 }, alignment: AlignmentType.JUSTIFIED, spacing: { before: 0, after: 60, line: LINE }, children: runs(it) }));
    case "numbered": return b.items.map(it => new Paragraph({ numbering: { reference: b.ref || "num1", level: 0 }, alignment: AlignmentType.JUSTIFIED, spacing: { before: 0, after: 60, line: LINE }, children: runs(it) }));
    case "eq": return [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 120, line: LINE }, children: runs(b.text, { size: 22 }) })];
    case "code": return b.lines.map(l => new Paragraph({ spacing: { before: 0, after: 0, line: 252 }, indent: { left: 360 }, shading: { type: ShadingType.CLEAR, fill: "F2F2F2" }, children: [new TextRun({ text: l, font: "Courier New", size: 19 })] }));
    case "center": return [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 0, after: b.after ?? 120, line: LINE }, children: runs(b.text, { size: (b.size || 12) * 2, bold: !!b.bold }) })];
    case "left": return [new Paragraph({ alignment: AlignmentType.LEFT, spacing: { before: 0, after: b.after ?? 60, line: LINE }, children: runs(b.text, { size: 24, bold: !!b.bold }) })];
    case "spacer": return [new Paragraph({ spacing: { before: 0, after: 0, line: LINE }, children: [new TextRun({ text: "", font: FONT })] })].concat(Array.from({ length: (b.n || 1) - 1 }, () => new Paragraph({ spacing: { before: 0, after: 0, line: LINE }, children: [] })));
    case "pagebreak": return [new Paragraph({ children: [new PageBreak()] })];
    case "frontheading": return [new Paragraph({ style: "FrontHeading", alignment: AlignmentType.CENTER, pageBreakBefore: b.nobreak ? false : true, children: [new TextRun({ text: b.text, font: FONT })] })];
    case "toc": return [new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" })];
    case "lof": return [new TableOfContents("List of Figures", { hyperlink: true, captionLabel: "Figure" })];
    case "lot": return [new TableOfContents("List of Tables", { hyperlink: true, captionLabel: "Table" })];
    case "fig": {
      const f = path.join(FIG_DIR, b.file); const { w, h } = pngSize(f);
      const maxW = b.width || 600; const width = Math.min(maxW, 600); const height = Math.round(width * h / w);
      const maxH = 640; let W = width, H = height; if (H > maxH) { W = Math.round(W * maxH / H); H = maxH; }
      return [new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 0, line: 240 },
        children: [new ImageRun({ type: "png", data: fs.readFileSync(f), transformation: { width: W, height: H }, altText: { title: b.caption, description: b.caption, name: b.file } })] }), caption("Figure", b.caption)];
    }
    case "table": return table(b);
    default: throw new Error("unknown block " + b.t);
  }
}

function table(b) {
  const n = b.header.length; const total = TEXT_W;
  let cw = b.widths ? b.widths.map(x => Math.round(x * total / b.widths.reduce((a, c) => a + c, 0))) : Array(n).fill(Math.floor(total / n));
  cw[cw.length - 1] += total - cw.reduce((a, c) => a + c, 0);
  const sz = b.size || 19; const bd = { style: BorderStyle.SINGLE, size: 4, color: "7F7F7F" }; const borders = { top: bd, bottom: bd, left: bd, right: bd };
  const cell = (txt, i, head, shade) => new TableCell({ width: { size: cw[i], type: WidthType.DXA }, borders, verticalAlign: VerticalAlign.CENTER,
    shading: head ? { type: ShadingType.CLEAR, fill: "D9E2F3" } : (shade ? { type: ShadingType.CLEAR, fill: "F7F7F7" } : undefined),
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({ alignment: (head || (b.align && b.align[i] === "l")) ? AlignmentType.LEFT : (b.align && b.align[i] === "c" ? AlignmentType.CENTER : AlignmentType.RIGHT), spacing: { before: 0, after: 0, line: 252 },
      children: runs(String(txt), { size: sz, bold: head }) })] });
  const rows = [new TableRow({ tableHeader: true, cantSplit: true, children: b.header.map((h, i) => cell(h, i, true)) })]
    .concat(b.rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, i, false, ri % 2 === 1)) })));
  const out = [caption("Table", b.caption), new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: cw, rows })];
  if (b.note) out.push(new Paragraph({ spacing: { before: 60, after: 160, line: 240 }, children: runs(b.note, { size: 20, italics: true }) }));
  else out.push(new Paragraph({ spacing: { before: 0, after: 160, line: 240 }, children: [] }));
  return out;
}

const footer = (fmt, start) => ({ default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 22 })] })] }) });
const sections = content.sections.map(s => ({
  properties: { page: { ...PAGE, pageNumbers: s.numbering ? { start: s.numbering.start, formatType: s.numbering.fmt === "roman" ? NumberFormat.LOWER_ROMAN : NumberFormat.DECIMAL } : undefined } },
  footers: s.numbering ? footer() : undefined,
  children: s.blocks.flatMap(block),
}));

const doc = new Document({
  creator: "Student (placeholder)", title: content.title, description: "Predictive Analysis CA2 project report",
  styles: {
    default: { document: { run: { font: FONT, size: 24 }, paragraph: { spacing: { line: LINE } } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 28, bold: true, color: "000000" }, paragraph: { spacing: { before: 240, after: 200, line: LINE }, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 26, bold: true, color: "000000" }, paragraph: { spacing: { before: 240, after: 120, line: LINE }, outlineLevel: 1, keepNext: true } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 24, bold: true, italics: true, color: "000000" }, paragraph: { spacing: { before: 200, after: 100, line: LINE }, outlineLevel: 2, keepNext: true } },
      { id: "FrontHeading", name: "Front Heading", basedOn: "Normal", next: "Normal", run: { font: FONT, size: 28, bold: true, color: "000000" }, paragraph: { spacing: { before: 240, after: 200, line: LINE }, keepNext: true } },
      { id: "TOC1", name: "toc 1", basedOn: "Normal", next: "Normal", run: { font: FONT, size: 24, bold: true }, paragraph: { spacing: { before: 60, after: 20, line: 276 } } },
      { id: "TOC2", name: "toc 2", basedOn: "Normal", next: "Normal", run: { font: FONT, size: 24 }, paragraph: { spacing: { before: 0, after: 0, line: 276 }, indent: { left: 360 } } },
      { id: "TableofFigures", name: "table of figures", basedOn: "Normal", next: "Normal", run: { font: FONT, size: 22 }, paragraph: { spacing: { before: 0, after: 60, line: 252 } } },
      { id: "Caption", name: "caption", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 22, color: "000000" } },
    ],
  },
  numbering: { config: [
    { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ...["num1", "num2", "num3", "num4", "num5", "num6", "num7", "num8", "num9"].map(r => ({ reference: r, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] })),
  ] },
  sections,
});
Packer.toBuffer(doc).then(buf => { fs.writeFileSync(process.argv[2] || path.join(__dirname, "report.docx"), buf); console.log("wrote", buf.length, "bytes"); });

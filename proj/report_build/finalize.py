"""Post-process the DOCX: cache SEQ caption numbers and fill the TOC / List of Figures / List of Tables
with real page numbers (measured from a LibreOffice PDF render). Word can still refresh them (right-click -> Update field)."""
import json, re, subprocess, sys, zipfile, shutil, html
from pathlib import Path

here = Path(__file__).parent
src, out = Path(sys.argv[1]), Path(sys.argv[2])
content = json.load(open(here / "content.json"))
main = content["sections"][2]["blocks"]
tmp = Path("/tmp/fin"); shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir()

def read_zip(p):
    with zipfile.ZipFile(p) as z: return {n: z.read(n) for n in z.namelist()}
def write_zip(files, p):
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in files.items(): z.writestr(n, b)
def esc(t): return html.escape(t, quote=False)

files = read_zip(src); xml = files["word/document.xml"].decode("utf8")
# ---- 1. cache SEQ numbers ------------------------------------------------------------------------
cnt = {"Figure": 0, "Table": 0}
def seq(m):
    lab = m.group(1); cnt[lab] += 1
    return (f'<w:r><w:fldChar w:fldCharType="begin"/><w:instrText xml:space="preserve">SEQ {lab}</w:instrText><w:fldChar w:fldCharType="separate"/></w:r>'
            f'<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/><w:b/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr><w:t>{cnt[lab]}</w:t></w:r>'
            f'<w:r><w:fldChar w:fldCharType="end"/></w:r>')
xml = re.sub(r'<w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/><w:instrText xml:space="preserve">SEQ (Figure|Table)</w:instrText><w:fldChar w:fldCharType="separate"/><w:fldChar w:fldCharType="end"/></w:r>', seq, xml)
xml = xml.replace('TOC \\a &quot;Figure&quot; \\h', 'TOC \\h \\c &quot;Figure&quot;').replace('TOC \\a &quot;Table&quot; \\h', 'TOC \\h \\c &quot;Table&quot;')
def fix_rule(x): return re.sub(r'w:line="(\d+)"(?![^>]*w:lineRule)', r'w:line="\1" w:lineRule="auto"', x)
xml = fix_rule(xml)
files["word/document.xml"] = xml.encode("utf8")
files["word/styles.xml"] = fix_rule(files["word/styles.xml"].decode("utf8")).encode("utf8")
write_zip(files, tmp / "pass1.docx")
subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(tmp), str(tmp / "pass1.docx")], capture_output=True)
txt = subprocess.run(["pdftotext", "-layout", str(tmp / "pass1.pdf"), "-"], capture_output=True, text=True).stdout
pages = [re.sub(r"\s+", " ", p) for p in txt.split("\f")]
norm = lambda s: re.sub(r"\s+", " ", s.replace("’", "'")).strip()
pages = [p.replace("’", "'") for p in pages]
start = next(i for i, p in enumerate(pages) if "1. Introduction" in p and "TABLE OF CONTENTS" not in p)
print("physical pages:", len(pages) - 1, "| main section starts at physical page", start + 1)

# ---- 2. entries ---------------------------------------------------------------------------------
heads = [(b["t"], b["text"]) for b in main if b["t"] in ("h1", "h2")]
figs = [(f"Figure {i+1}: {b['caption']}", f"Figure {i+1}:") for i, b in enumerate([x for x in main if x["t"] == "fig"])]
tabs = [(f"Table {i+1}: {b['caption']}", f"Table {i+1}:") for i, b in enumerate([x for x in main if x["t"] == "table"])]
def locate(key, ptr):
    k = norm(key)[:45]
    for i in range(ptr, len(pages)):
        if k in pages[i]: return i
    for i in range(start, len(pages)):          # fallback: anywhere in main
        if k in pages[i]: return i
    return None
ptr, toc_e = start, []
for lvl, t in heads:
    i = locate(t, ptr); ptr = i if i is not None else ptr
    toc_e.append((lvl, t, (i - start + 1) if i is not None else "?"))
fig_e = [(full, locate(key, start)) for full, key in figs]; fig_e = [(f, (i - start + 1) if i is not None else "?") for f, i in fig_e]
tab_e = [(full, locate(key, start)) for full, key in tabs]; tab_e = [(f, (i - start + 1) if i is not None else "?") for f, i in tab_e]
print("unlocated:", sum(1 for e in toc_e if e[2] == "?"), sum(1 for e in fig_e if e[1] == "?"), sum(1 for e in tab_e if e[1] == "?"))

def para(style, text, page, first_instr=None, last=False, indent=None):
    pre = (f'<w:r><w:fldChar w:fldCharType="begin"/><w:instrText xml:space="preserve">{first_instr}</w:instrText><w:fldChar w:fldCharType="separate"/></w:r>' if first_instr else "")
    post = '<w:r><w:fldChar w:fldCharType="end"/></w:r>' if last else ""
    return (f'<w:p><w:pPr><w:pStyle w:val="{style}"/><w:tabs><w:tab w:val="right" w:leader="dot" w:pos="9016"/></w:tabs></w:pPr>{pre}'
            f'<w:r><w:t xml:space="preserve">{esc(text)}</w:t></w:r><w:r><w:tab/></w:r><w:r><w:t>{page}</w:t></w:r>{post}</w:p>')
def build(entries, style_of, instr):
    ps = []
    for i, (style, text, page) in enumerate(entries):
        ps.append(para(style, text, page, instr if i == 0 else None, i == len(entries) - 1))
    return "".join(ps)
toc_xml = build([("TOC1" if l == "h1" else "TOC2", t, p) for l, t, p in toc_e], None, 'TOC \\h \\o &quot;1-2&quot;')
fig_xml = build([("TableofFigures", f, p) for f, p in fig_e], None, 'TOC \\h \\c &quot;Figure&quot;')
tab_xml = build([("TableofFigures", f, p) for f, p in tab_e], None, 'TOC \\h \\c &quot;Table&quot;')
xml = files["word/document.xml"].decode("utf8")
def fill(xml, alias, body):
    pat = re.compile(r'(<w:sdtPr><w:alias w:val="' + alias + r'"/></w:sdtPr><w:sdtContent>)(.*?)(</w:sdtContent>)', re.S)
    assert pat.search(xml), alias
    return pat.sub(lambda m: m.group(1) + body + m.group(3), xml, count=1)
xml = fill(xml, "Table of Contents", toc_xml); xml = fill(xml, "List of Figures", fig_xml); xml = fill(xml, "List of Tables", tab_xml)
xml = fix_rule(xml.replace(' w:dirty="true"', ""))
files["word/document.xml"] = xml.encode("utf8")
write_zip(files, out); print("final docx:", out)

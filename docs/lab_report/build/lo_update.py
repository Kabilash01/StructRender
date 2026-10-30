"""Open a .docx in headless LibreOffice, update every index (contents, lists of figures and
tables), then save the updated .docx and export a PDF; finally set every font in the saved .docx to Times New Roman
(LibreOffice adds its own default and theme fonts on save).  Usage: python3 lo_update.py in.docx out.docx out.pdf"""
import os, re, shutil, subprocess, sys, tempfile, time, zipfile
import uno
from com.sun.star.beans import PropertyValue

def pv(n, v):
    p = PropertyValue(); p.Name = n; p.Value = v; return p

src, out_docx, out_pdf = (os.path.abspath(a) for a in sys.argv[1:4])
prof = tempfile.mkdtemp(prefix="lo_prof_")
pipe = "lab_report_" + str(os.getpid())
proc = subprocess.Popen(["soffice", f"-env:UserInstallation=file://{prof}", "--headless", "--invisible", "--norestore",
                         f"--accept=pipe,name={pipe};urp;"], env={**os.environ, "SAL_USE_VCLPLUGIN": "svp"})
try:
    local = uno.getComponentContext()
    resolver = local.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", local)
    for _ in range(60):
        try:
            ctx = resolver.resolve(f"uno:pipe,name={pipe};urp;StarOffice.ComponentContext"); break
        except Exception:
            time.sleep(0.5)
    desk = ctx.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
    doc = desk.loadComponentFromURL(uno.systemPathToFileUrl(src), "_blank", 0, (pv("Hidden", True),))
    idx = doc.getDocumentIndexes()
    for _ in range(2):                      # twice: the lists' own length moves later page numbers
        for i in range(idx.getCount()):
            idx.getByIndex(i).update()
        doc.getTextFields().refresh()
    doc.storeToURL(uno.systemPathToFileUrl(out_docx), (pv("FilterName", "MS Word 2007 XML"),))
    doc.storeToURL(uno.systemPathToFileUrl(out_pdf), (pv("FilterName", "writer_pdf_Export"),))
    doc.close(True)
finally:
    try:
        desk.terminate()
    except Exception:
        pass
    time.sleep(1); proc.kill()

TNR = "Times New Roman"
def _tnr(xml):
    xml = re.sub(r"<w:rFonts\b[^>]*/>", f'<w:rFonts w:ascii="{TNR}" w:hAnsi="{TNR}" w:cs="{TNR}" w:eastAsia="{TNR}"/>', xml)
    xml = re.sub(r'(<a:(?:latin|ea|cs|font)\b[^>]*?typeface=")[^"]*"', rf'\g<1>{TNR}"', xml)
    return xml.replace('w:val="\uf0b7"', 'w:val="\u2022"')  # Symbol-font bullet -> plain bullet
tmp = out_docx + ".tmp"
with zipfile.ZipFile(out_docx) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename.endswith(".xml") and item.filename.startswith("word/"):
            data = _tnr(data.decode("utf-8")).encode("utf-8")
        zout.writestr(item, data)
shutil.move(tmp, out_docx)

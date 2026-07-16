"""
07_build_submission.py
======================
Package the three deliverables into submission-ready formats
(pdf / png / zip):

  submission/1_Writeup_Propeller_Performance.pdf   <- report/REPORT.md + figures
  submission/2_Screenshots.zip                     <- the 10 PNG figures
  submission/3_SourceCode.pdf                       <- all .py + .sql, readable
  submission/Propeller_Capstone_Full_Project.zip    <- whole runnable project

Rendering uses the pre-installed Chromium via Playwright (HTML -> PDF), so the
PDFs are self-contained (figures embedded as base64).
"""
import base64
import glob
import html
import os
import re
import zipfile

import markdown as md_lib
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPORT_MD = os.path.join(ROOT, "report", "REPORT.md")
FIG_DIR = os.path.join(ROOT, "outputs", "figures")
SUB = os.path.join(ROOT, "submission")
os.makedirs(SUB, exist_ok=True)

CHROMIUM = glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome")
CHROMIUM = CHROMIUM[0] if CHROMIUM else None

CSS = """
<style>
  @page { size: A4; margin: 18mm 16mm; }
  body { font-family: 'Helvetica Neue', Arial, sans-serif; color:#1a1a1a;
         font-size: 11.5px; line-height: 1.5; }
  h1 { font-size: 22px; color:#1F3A5F; border-bottom:3px solid #4C72B0;
       padding-bottom:6px; }
  h2 { font-size: 16px; color:#1F3A5F; margin-top:22px;
       border-bottom:1px solid #ccd; padding-bottom:3px; }
  h3 { font-size: 13px; color:#345; }
  table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size:10.5px; }
  th,td { border:1px solid #bbc; padding:5px 8px; text-align:left; }
  th { background:#eef2f8; }
  code { background:#f2f4f7; padding:1px 4px; border-radius:3px;
         font-family: 'SFMono-Regular', Consolas, monospace; font-size:10.5px; }
  pre { background:#f6f8fa; padding:10px; border-radius:6px; overflow-x:auto;
        border:1px solid #e1e4e8; }
  pre code { background:none; }
  blockquote { border-left:4px solid #DD8452; margin:10px 0; padding:4px 12px;
               background:#fff8f3; color:#443; }
  img.figure { width:100%; max-width:720px; display:block; margin:10px auto 4px;
               border:1px solid #ddd; border-radius:4px; }
  .cap { text-align:center; font-size:10px; color:#666; margin-bottom:16px; }
</style>
"""


def data_uri(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{b64}"


def build_report_html():
    with open(REPORT_MD) as f:
        text = f.read()
    body = md_lib.markdown(text, extensions=["tables", "fenced_code"])

    # Inline each figure right after its first mention, whether that mention is
    # bare (`NN_name.png`) or path-prefixed (`outputs/figures/NN_name.png`).
    figs = sorted(f for f in os.listdir(FIG_DIR) if f.endswith(".png"))
    embedded = set()
    for fig in figs:
        uri = data_uri(os.path.join(FIG_DIR, fig))
        img = (f'<img class="figure" src="{uri}"/>'
               f'<div class="cap">Figure &mdash; {fig}</div>')
        pattern = re.compile(r"(<code>(?:[^<]*/)?" + re.escape(fig) + r"</code>)")
        new_body, n = pattern.subn(r"\1" + img, body, count=1)
        if n:
            body = new_body
            embedded.add(fig)

    # Safety net: any figure not referenced inline is appended as a gallery so
    # every simulation screenshot appears in the PDF.
    missing = [f for f in figs if f not in embedded]
    if missing:
        gallery = ['<h2 style="page-break-before:always">Appendix &mdash; '
                   'additional figures</h2>']
        for fig in missing:
            uri = data_uri(os.path.join(FIG_DIR, fig))
            gallery.append(f'<img class="figure" src="{uri}"/>'
                           f'<div class="cap">Figure &mdash; {fig}</div>')
        body += "".join(gallery)

    return f"<!doctype html><html><head><meta charset='utf-8'>{CSS}</head>" \
           f"<body>{body}</body></html>"


def build_sourcecode_html():
    files = sorted(glob.glob(os.path.join(HERE, "*.py"))) + \
            [os.path.join(HERE, "sql", "week1_queries.sql")]
    order = ["config.py", "01_data_preparation.py", "02_solidity_analysis.py",
             "03_eda_visualizations.py", "04_ml_models.py", "05_run_sql.py",
             "06_tableau_dashboard.py", "07_build_submission.py", "run_all.py",
             "week1_queries.sql"]
    files = sorted(files, key=lambda p: order.index(os.path.basename(p))
                   if os.path.basename(p) in order else 99)

    parts = ["<h1>UAV Propeller Performance &mdash; Source Code</h1>",
             "<p>All analysis, machine-learning and SQL source for the capstone. "
             "Files are listed in execution order.</p>",
             "<h2>Contents</h2><ol>"]
    for p in files:
        parts.append(f"<li>{html.escape(os.path.basename(p))}</li>")
    parts.append("</ol>")
    for p in files:
        name = os.path.basename(p)
        with open(p) as f:
            code = f.read()
        parts.append(f'<h2 style="page-break-before:always">{html.escape(name)}</h2>')
        parts.append(f"<pre><code>{html.escape(code)}</code></pre>")
    return f"<!doctype html><html><head><meta charset='utf-8'>{CSS}</head>" \
           f"<body>{''.join(parts)}</body></html>"


def html_to_pdf(html_str, out_pdf):
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROMIUM)
        page = browser.new_page()
        page.set_content(html_str, wait_until="networkidle")
        page.pdf(path=out_pdf, format="A4", print_background=True,
                 margin={"top": "16mm", "bottom": "16mm",
                         "left": "14mm", "right": "14mm"})
        browser.close()


def zip_screenshots():
    out = os.path.join(SUB, "2_Screenshots.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for fig in sorted(os.listdir(FIG_DIR)):
            if fig.endswith(".png"):
                z.write(os.path.join(FIG_DIR, fig), fig)
        # include the propeller reference diagram too
        diag = os.path.join(ROOT, "report", "propeller_diagram.png")
        if os.path.exists(diag):
            z.write(diag, "00_propeller_reference_diagram.png")
    print("Wrote", out)


def zip_full_project():
    out = os.path.join(SUB, "Propeller_Capstone_Full_Project.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for base, _, names in os.walk(ROOT):
            if "/submission" in base or "/.git" in base:
                continue
            for n in names:
                fp = os.path.join(base, n)
                z.write(fp, os.path.relpath(fp, ROOT))
    print("Wrote", out)


def main():
    print("Rendering write-up PDF ...")
    html_to_pdf(build_report_html(),
                os.path.join(SUB, "1_Writeup_Propeller_Performance.pdf"))
    print("Rendering source-code PDF ...")
    html_to_pdf(build_sourcecode_html(),
                os.path.join(SUB, "3_SourceCode.pdf"))
    print("Zipping screenshots ...")
    zip_screenshots()
    print("Zipping full project ...")
    zip_full_project()
    print("\nSubmission files:")
    for f in sorted(os.listdir(SUB)):
        size = os.path.getsize(os.path.join(SUB, f)) / 1024
        print(f"  {f:45s} {size:8.1f} KB")


if __name__ == "__main__":
    main()

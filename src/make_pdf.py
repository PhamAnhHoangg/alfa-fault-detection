"""Builds docs/ALFA_fault_detection_documentation.pdf from the markdown documentation.

Needs pandoc and a LaTeX engine with xelatex (for example MiKTeX or TeX Live) on the PATH, and the DejaVu fonts.
Run from the repo root:  python src/make_pdf.py
The PDF puts the figure index and the open issues (limits and unsolved problems) first.
GIFs cannot play in a PDF, so tables that show a GIF next to a static picture keep only the static picture."""
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG = (ROOT / "figures").as_posix() + "/"
s = (ROOT / "docs" / "ALFA_fault_detection_documentation.md").read_text(encoding="utf-8-sig").replace("\r\n", "\n")
s = re.sub(r"\| Step by step \| (?:Static picture|Final figure) \|\n\|---\|---\|\n"
           r"\| !\[[^\]]*\]\([^)]*\.gif\) \| (!\[[^\]]*\]\([^)]*\.png\)) \|", r"\1", s)
s = re.sub(r"\[([^\]]+)\]\(\.\./figures/[^)]*\.gif\)", r"\1 (animated, in the repository)", s)
assert ".gif)" not in s, "a GIF is still linked; add its pattern above"
s = s.replace("](../figures/", "](" + FIG)
j, k = s.index("## Contents"), s.index("## Find the explanation")
s = s[:j] + s[k:]  # pandoc builds its own contents list
issues = s[s.index("### Limits and open questions"):s.index("### Credits")]
issues = issues.replace("### Limits and open questions", "#### Limits").replace("### Unsolved problems", "#### Unsolved problems")
i = s.index("## Part 1.")
s = s[:i] + "## Open issues\n\nCopied from Part 1 so they come first.\n\n" + issues + "\n" + s[i:]

HEAD = r"""\usepackage{fvextra}
\fvset{breaklines=true}
\setkeys{Gin}{width=\linewidth,height=0.42\textheight,keepaspectratio}
\usepackage{xurl}
\renewcommand{\arraystretch}{1.15}
\usepackage{float}
\floatplacement{figure}{H}
\setlength{\parindent}{0pt}
"""
with tempfile.TemporaryDirectory() as tmp:
    md, head = Path(tmp) / "doc.md", Path(tmp) / "head.tex"
    md.write_text(s, encoding="utf-8")
    head.write_text(HEAD, encoding="utf-8")
    out = ROOT / "docs" / "ALFA_fault_detection_documentation.pdf"
    subprocess.run(["pandoc", str(md), "-f",
                    "markdown-tex_math_single_backslash-raw_tex-raw_attribute+tex_math_dollars-subscript-superscript",
                    "-o", str(out), "--pdf-engine=xelatex", "-H", str(head), "--columns=40",
                    "--shift-heading-level-by=-1", "--toc", "--toc-depth=1",
                    "-V", "geometry:margin=2cm", "-V", "fontsize=10pt", "-V", "mainfont=DejaVu Sans",
                    "-V", "monofont=DejaVu Sans Mono", "-V", "colorlinks=true", "-V", "linkcolor=blue",
                    "-V", "urlcolor=blue"], check=True)
print("saved", out)

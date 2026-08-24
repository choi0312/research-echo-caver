# Paper source

`main.tex` is an arXiv-oriented article source. `references.bib` contains the
primary JUDO, MMAD, LoRA, O-LoRA, and FREB-CAVER references. The committed PDF
is rendered from the same sealed metrics by `../scripts/build_paper_pdf.py` so
that the archive remains buildable even without a local TeX distribution.

With a standard TeX Live installation:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The ReportLab artifact can be regenerated from the repository root:

```bash
python3 scripts/build_paper_pdf.py
```


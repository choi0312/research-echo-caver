# Paper source

`main.tex` is an arXiv-oriented article source. `references.bib` contains the
primary JUDO, AD-Copilot, MMAD, LoRA, O-LoRA, and FREB-CAVER references. The
generated PDF and presentation figures are intentionally not versioned in this
minimal reproducibility release.

With a standard TeX Live installation:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Generated build products should be kept outside the source repository.

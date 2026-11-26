# IEEE Research Paper: Hybrid VM Placement Optimization

This folder contains the complete, publication-ready LaTeX source files formatted according to the official **IEEE Conference Proceedings (IEEEtran)** specifications.

## Files
- `main.tex`: Full IEEEtran LaTeX source code including title, authors, abstract, keywords, all 7 technical sections, mathematical formulation, equations, pseudocode, and performance tables.
- `references.bib`: Complete BibTeX reference database with peer-reviewed citations from IEEE, ACM, Elsevier, and Science.

## Compilation Instructions

### Using Local LaTeX Installation
To compile the document into a publication-ready PDF, run:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or using `latexmk`:
```bash
latexmk -pdf main.tex
```

### Using Overleaf / Web LaTeX Editor
1. Create a new blank project on [Overleaf](https://www.overleaf.com/).
2. Upload `main.tex` and `references.bib`.
3. Set the compiler to **pdfLaTeX** and press **Recompile**.

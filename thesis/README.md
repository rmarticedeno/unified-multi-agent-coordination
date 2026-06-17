# Thesis

This directory contains the LaTeX source for the master's thesis associated with the project.

## Structure

- `main.tex`: root LaTeX document.
- `chapters/`: chapter source files.
- `figures/`: figures and diagrams used by the thesis.
- `references/references.bib`: BibTeX bibliography database.
- `build/`: generated PDFs and LaTeX build output.

## Build

From this directory:

```powershell
New-Item -ItemType Directory -Force build | Out-Null
pdflatex -output-directory=build main.tex
bibtex build/main
pdflatex -output-directory=build main.tex
pdflatex -output-directory=build main.tex
```

Generated build output is intentionally ignored by Git.

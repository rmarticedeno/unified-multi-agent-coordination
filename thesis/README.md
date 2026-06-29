# Thesis

This directory contains the LaTeX source for the master's thesis associated with the project.

## Structure

- `main.tex`: root LaTeX document.
- `uhmasterthesis.cls`: University of Havana branded thesis class.
- `Graphics/uhlogo.pdf`: University of Havana cover mark supplied by the branding template.
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

## Argument structure

The thesis follows this research sequence: introduction and problem formulation; critical state of the art; research methodology; proposed protocol-independent theoretical framework; protocol-adaptable system design; evaluation and results; discussion; and conclusions. The evaluation chapter currently defines the reporting protocol and explicitly marks empirical measurements as pending so that unexecuted experiments are not presented as findings.

The cover follows the University of Havana and Faculty of Mathematics and Computer Science branding supplied in the curated template. Its QR code is generated at compile time from the repository URL declared with `\repositoryurl{...}` in `main.tex`; no separately generated QR image is required.

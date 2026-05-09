arXiv Submission Package
========================

Paper:    Experience-Dependent Structural Plasticity in Knowledge Graphs
Category: cs.AI
Main file: plasticity-paper.tex

Build:
  pdflatex plasticity-paper.tex
  bibtex plasticity-paper
  pdflatex plasticity-paper.tex
  pdflatex plasticity-paper.tex

Pre-submission notes:
  Submit after flagship.

After building, update shared/ imports in the .tex file to local paths:
  \usepackage{cerebrum-macros} (already in submission dir)
  \input{notation} (already in submission dir)
  \input{author-block} (already in submission dir)

Replace all [CEREBRUM_REPORT_PLACEHOLDER] / [CEREBRUM_REPORT_ID] with
the actual arXiv ID using: python scripts/update_arxiv_papers.py --report-id XXXXX

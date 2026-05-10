arXiv Submission Package
========================

Paper:    Holographic Indexing: Privacy-Preserving Discovery in Federated KG Networks
Category: cs.DC
Main file: holographic-indexing.tex

Build:
  pdflatex holographic-indexing.tex
  bibtex holographic-indexing
  pdflatex holographic-indexing.tex
  pdflatex holographic-indexing.tex

Pre-submission notes:
  Submit after flagship.

After building, update shared/ imports in the .tex file to local paths:
  \usepackage{cerebrum-macros} (already in submission dir)
  \input{notation} (already in submission dir)
  \input{author-block} (already in submission dir)

Replace all [CEREBRUM_REPORT_PLACEHOLDER] / [CEREBRUM_REPORT_ID] with
the actual arXiv ID using: python scripts/update_arxiv_papers.py --report-id XXXXX

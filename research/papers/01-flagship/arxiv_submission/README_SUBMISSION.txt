arXiv Submission Package
========================

Paper:    CEREBRUM: Training-Free KG Reasoning via Community-Structured Graph Attention
Category: cs.AI
Main file: cerebrum-flagship.tex

Build:
  pdflatex cerebrum-flagship.tex
  bibtex cerebrum-flagship
  pdflatex cerebrum-flagship.tex
  pdflatex cerebrum-flagship.tex

Pre-submission notes:
  Submit second. Replace [CEREBRUM_REPORT_ID] with Technical Report arXiv ID first.

After building, update shared/ imports in the .tex file to local paths:
  \usepackage{cerebrum-macros} (already in submission dir)
  \input{notation} (already in submission dir)
  \input{author-block} (already in submission dir)

Replace all [CEREBRUM_REPORT_PLACEHOLDER] / [CEREBRUM_REPORT_ID] with
the actual arXiv ID using: python scripts/update_arxiv_papers.py --report-id XXXXX

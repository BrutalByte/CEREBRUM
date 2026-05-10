arXiv Submission Package
========================

Paper:    Production Knowledge Graph Reasoning: Fault Tolerance, Streaming, Maintenance
Category: cs.SE
Main file: production-kg.tex

Build:
  pdflatex production-kg.tex
  bibtex production-kg
  pdflatex production-kg.tex
  pdflatex production-kg.tex

Pre-submission notes:
  Submit after flagship.

After building, update shared/ imports in the .tex file to local paths:
  \usepackage{cerebrum-macros} (already in submission dir)
  \input{notation} (already in submission dir)
  \input{author-block} (already in submission dir)

Replace all [CEREBRUM_REPORT_PLACEHOLDER] / [CEREBRUM_REPORT_ID] with
the actual arXiv ID using: python scripts/update_arxiv_papers.py --report-id XXXXX

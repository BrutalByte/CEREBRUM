arXiv Submission Package
========================

Paper:    CEREBRUM v2.52: Complete Technical Specification
Category: cs.AI
Main file: cerebrum-v252-report.tex

Build:
  pdflatex cerebrum-v252-report.tex
  bibtex cerebrum-v252-report
  pdflatex cerebrum-v252-report.tex
  pdflatex cerebrum-v252-report.tex

Pre-submission notes:
  Submit first — other papers reference this arXiv ID.

After building, update shared/ imports in the .tex file to local paths:
  \usepackage{cerebrum-macros} (already in submission dir)
  \input{notation} (already in submission dir)
  \input{author-block} (already in submission dir)

Replace all [CEREBRUM_REPORT_PLACEHOLDER] / [CEREBRUM_REPORT_ID] with
the actual arXiv ID using: python scripts/update_arxiv_papers.py --report-id XXXXX

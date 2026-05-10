arXiv Submission Package
========================

Paper:    Triple-Signal Consensus: Temperature-Annealed Community Detection
Category: cs.SI
Main file: tsc-paper.tex

Build:
  pdflatex tsc-paper.tex
  bibtex tsc-paper
  pdflatex tsc-paper.tex
  pdflatex tsc-paper.tex

Pre-submission notes:
  Fill in LFR benchmark values (run scripts/run_lfr_benchmark.py) before submitting.

After building, update shared/ imports in the .tex file to local paths:
  \usepackage{cerebrum-macros} (already in submission dir)
  \input{notation} (already in submission dir)
  \input{author-block} (already in submission dir)

Replace all [CEREBRUM_REPORT_PLACEHOLDER] / [CEREBRUM_REPORT_ID] with
the actual arXiv ID using: python scripts/update_arxiv_papers.py --report-id XXXXX

arXiv Submission Package
========================

Paper:    Schema-Derived Relation Boost and Principled Hyperparameter Initialization
          for Training-Free Multi-Hop Knowledge Graph Reasoning
Category: cs.AI  (cross-list: cs.IR, cs.LG)
Main file: sdrb-paper.tex

Build:
  pdflatex sdrb-paper.tex
  bibtex sdrb-paper
  pdflatex sdrb-paper.tex
  pdflatex sdrb-paper.tex

Pre-submission checklist:
  [ ] Phase 204 tuner validated on full 14,274 questions — update Table 1 H@1 if > 58.9%
  [ ] Phase 205 ParameterInitializer built and tested — confirms Section 8 formulas
  [ ] Hetionet cross-KB validation complete — fills in typed_heterogeneous row in Table 4
  [ ] Replace [CEREBRUM_REPORT_ID] with Technical Report arXiv ID
        python ../../scripts/update_arxiv_papers.py --report-id XXXXX
  [ ] Build PDF and verify two-column layout, table formatting, equation numbering
  [ ] Check all \todo{} and \note{} macros are removed (none currently in draft)

Submit after:
  1. Technical Report (00-technical-report) submitted first — provides \CEREBRUMReportID
  2. This paper second (or simultaneously if independent citation is acceptable)

arXiv metadata:
  Title:    Schema-Derived Relation Boost and Principled Hyperparameter Initialization
            for Training-Free Multi-Hop Knowledge Graph Reasoning
  Authors:  Bryan Alexander Buchorn
  Abstract: [use abstract from sdrb-paper.tex]
  Primary:  cs.AI
  Cross:    cs.IR, cs.LG
  Comments: 8 pages, 5 tables. Code: https://github.com/BrutalByte/CEREBRUM

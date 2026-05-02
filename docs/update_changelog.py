import os

def update_changelog():
    path = '../CHANGELOG.md'
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_entry = """## [2.35.0] - 2026-04-29
### Added
- **Phase 150: Frontal Engine Executive Strategy**: Autonomous reasoning orchestration.
  - Implemented `FrontalEngine` for dynamic selection of reasoning strategies (FAST, HYBRID, DEEP).
  - Integrated `ResearchAgent` coupling to trigger targeted KG discovery when epistemic gaps are detected.
  - Added `epistemic_gaps` tracking in `BeamTraversal` to identify "grounding-starved" paths.
- **Phase 149: Cingulate Engine (Reasoning Verifier)**: Autonomous hub-flooding detection.
  - Implemented `ProvenanceValidator` to detect "hub-flooding" signatures in reasoning paths.
  - Added recursive refinement loop in `CerebrumGraph.query()` to retry with stricter constraints on failure.
  - Stabilized 3-hop MetaQA ranking by pruning high-entropy noise.

"""
    
    # Insert after the header
    header_end = content.find('Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).')
    if header_end != -1:
        header_end += len('Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).')
        content = content[:header_end] + "\n\n" + new_entry + content[header_end+2:]
    
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(content)
    print("Successfully updated CHANGELOG.md")

if __name__ == "__main__":
    update_changelog()

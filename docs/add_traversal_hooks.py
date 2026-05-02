import os
import re

def modify_traversal():
    path = '../reasoning/traversal.py'
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Add epistemic_gaps to __init__
    if 'self.epistemic_gaps = []' not in content:
        content = content.replace('self.quantized = quantized', 'self.quantized = quantized\n        self.epistemic_gaps = []  # Phase 150')

    # 2. Add gap detection to _prune_candidates
    gap_logic = """
        # Phase 150: Detect Epistemic Gaps (Frontal Engine)
        if hop > 1 and candidates and not any(p.score > 0.4 for p in candidates):
            # Identify the top potential but "grounding-starved" paths
            for p in sorted(candidates, key=lambda x: x.score, reverse=True)[:2]:
                self.epistemic_gaps.append({
                    "source": p.nodes[0],
                    "target": p.tail,
                    "score": p.score,
                    "hop": hop
                })
"""
    if 'Phase 150: Detect Epistemic Gaps' not in content:
        content = content.replace('        if self.lateral_inhibition_ratio > 0.0:', gap_logic + '        if self.lateral_inhibition_ratio > 0.0:')

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    modify_traversal()
    print("Traversal hooks added.")

import os

def update_paper():
    path = 'CEREBRUM_MASTER_PAPER.md'
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Update version and status in headers
    content = content.replace('v2.30.0 (Phase 149 (Cingulate Engine) COMPLETE)', 'v2.35.0 (Phase 150 (Frontal Engine) COMPLETE)')
    
    # 2. Add Recent Advances section before Conclusion
    if '### 4. Conclusion' in content:
        recent_advances = """### 3.5 The Executive Mind: Frontal and Cingulate Engines (Phases 149-150)
In v2.35.0, CEREBRUM moves beyond simple traversal toward executive orchestration. The **Frontal Engine** (Phase 150) implements a meta-reasoning layer that analyzes candidate paths and dynamically selects between FAST (traversal only), HYBRID (async research), and DEEP (suspend for research) strategies. This is coupled with the **Cingulate Engine** (Phase 149), which monitors reasoning entropy and detects "hub-flooding" signatures—situations where a few high-degree nodes overwhelm the beam. When such flooding is detected, the Cingulate Engine triggers a recursive refinement loop, retrying the query with stricter pruning constraints to recover signal from the noise.

"""
        content = content.replace('### 4. Conclusion', recent_advances + '### 4. Conclusion')

    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(content)
    print("Successfully updated CEREBRUM_MASTER_PAPER.md")

if __name__ == "__main__":
    update_paper()

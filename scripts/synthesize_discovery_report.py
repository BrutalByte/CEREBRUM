"""
SynthesizeDiscoveryReport: Aggregates autonomous research findings into a domain-agnostic 
verification format (JSON) and a human-readable summary (Markdown).
"""
import json
from core.candidate_registry import CandidateRegistry
from core.provenance_ledger import ProvenanceLedger

def synthesize_report(json_output: str, md_output: str):
    registry = CandidateRegistry()
    ledger = ProvenanceLedger()
    
    report = {
        "metadata": {"version": "2.51.0", "type": "DiscoveryVerificationReport"},
        "discoveries": []
    }
    
    # Extract entries from registry
    # Note: Using the internal _entries structure directly for domain-agnostic access
    for node_id, entry in registry._entries.items():
        discovery = {
            "node_id": node_id,
            "confidence": entry.get("confidence", 0.0),
            "reasoning_path": entry.get("path", []),
            "community_provenance": entry.get("community_id", "unknown"),
            "validation_status": entry.get("status", "pending")
        }
        report["discoveries"].append(discovery)
        
    # 1. Save JSON for programmatic audit
    with open(json_output, 'w') as f:
        json.dump(report, f, indent=2)
        
    # 2. Generate Human-Readable Markdown Report
    with open(md_output, 'w') as f:
        f.write("# Discovery Verification Report (v2.52.0)\n\n")
        f.write(f"**Total Findings**: {len(report['discoveries'])}\n\n")
        for i, d in enumerate(report['discoveries']):
            f.write(f"## Finding {i+1}: {d['node_id']}\n")
            f.write(f"- **Confidence**: {d['confidence']:.4f}\n")
            f.write(f"- **Provenance**: Community {d['community_provenance']}\n")
            f.write(f"- **Reasoning Trace**: `{' -> '.join(d['reasoning_path'])}`\n\n")
        
    print(f"Reports generated: {json_output} (JSON) and {md_output} (MD)")

if __name__ == "__main__":
    synthesize_report("discovery_verification_report.json", "discovery_verification_report.md")

"""
LARQL to CEREBRUM Transpiler.

Extracts highly confident neural relations from a LARQL VIndex and
exports them into a CEREBRUM-compliant CSV format for ingestion.
"""

import csv
import logging
import argparse
from typing import Dict, List, Optional
from core.larql_client import LarqlClient

# Map common LARQL/LLM neural labels to CEREBRUM canonical relations
RELATION_MAP = {
    "INFLUENCED": "INFLUENCED",
    "TREATS": "TREATS",
    "CAUSES": "CAUSES",
    "PART_OF": "PART_OF",
    "RELATED_TO": "RELATED_TO",
    "MEMBER_OF": "MEMBER_OF"
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("larql_transpiler")

def transpile(
    client: LarqlClient, 
    nodes: List[str], 
    output_path: str, 
    confidence_threshold: float = 0.85
):
    """
    Query neural weights for top-k connections and export to CSV.
    """
    count = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "relation", "target", "confidence"])
        
        for node in nodes:
            links = client.find_neural_neighbors(node, top_k=20)
            for link in links:
                if link.confidence >= confidence_threshold:
                    # Normalise relation
                    rel = RELATION_MAP.get(link.relation.upper(), "RELATED_TO")
                    
                    writer.writerow([link.source, rel, link.target, link.confidence])
                    count += 1
    
    logger.info("Transpilation complete: Exported %d high-confidence edges to %s", count, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transpile LARQL weights to CEREBRUM CSV")
    parser.add_argument("--endpoint", help="LARQL API endpoint")
    parser.add_argument("--vindex", help="Path to local VIndex")
    parser.add_argument("--output", default="neural_knowledge.csv")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    client = LarqlClient(endpoint=args.endpoint, vindex_path=args.vindex)
    
    # In a real run, we'd pull these from the existing adapter
    # For now, placeholder for entities
    nodes = ["newton", "einstein", "caffeine"]
    
    transpile(client, nodes, args.output, args.threshold)

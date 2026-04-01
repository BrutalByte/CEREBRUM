import bz2
import json
import csv
import os

# CEREBRUM: Hetionet Medical Subset Extractor (v1.4.0)
# Extracts a 100k edge subset from the 2.25M edge Hetionet graph.

INPUT_BZ2  = 'benchmarks/data/hetionet/hetionet-v1.0.json.bz2'
OUTPUT_CSV = 'tests/fixtures/hetionet_medical_100k.csv'
MAX_EDGES  = 100000

def extract():
    if not os.path.exists(INPUT_BZ2):
        print(f"❌ Error: {INPUT_BZ2} not found. Run benchmark first to download.")
        return

    print(f"Opening Hetionet ({INPUT_BZ2})...")
    
    with bz2.open(INPUT_BZ2, 'rt') as f:
        data = json.load(f)
        
    nodes = {n['kind'] + '::' + n['name']: n['name'] for n in data['nodes']}
    # Map internal IDs to readable names
    id_to_name = {n['identifier']: n['name'] for n in data['nodes']}
    
    edges = data['edges']
    print(f"Total edges in full set: {len(edges)}")
    
    # Take a 100k subset
    subset = edges[:MAX_EDGES]
    
    print(f"Writing {len(subset)} edges to {OUTPUT_CSV}...")
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["source", "target", "relation"])
        
        for e in subset:
            # Note: identifiers are used here, we'll map them if possible
            # But the 'source_id' and 'target_id' are indices/identifiers
            # We'll use names for better reasoning traces.
            # Hetionet JSON structure: {source_id: [...], target_id: [...], kind: "..."}
            src = id_to_name.get(e['source_id'][1], str(e['source_id'][1]))
            tgt = id_to_name.get(e['target_id'][1], str(e['target_id'][1]))
            rel = e['kind']
            writer.writerow([src, tgt, rel])
            
    print(f"✅ Success: Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    extract()

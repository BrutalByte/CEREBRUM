import os
import csv

# CEREBRUM: MetaQA KB Converter (v1.4.0)
# Converts the raw pipe-separated kb.txt into a Studio-compatible CSV.

INPUT_FILE  = 'benchmarks/data/metaqa/kb.txt'
OUTPUT_FILE = 'tests/fixtures/metaqa_movies.csv'

def convert():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found.")
        return
        
    print(f"Converting MetaQA KB ({INPUT_FILE})...")
    
    count = 0
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in:
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(["source", "relation", "target"])
            
            for line in f_in:
                parts = line.strip().split('|')
                if len(parts) == 3:
                    writer.writerow(parts)
                    count += 1
                    
    print(f"✅ Success: Converted {count} triples.")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    convert()

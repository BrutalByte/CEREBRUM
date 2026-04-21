import os
import sys
import subprocess
import time
import shutil
from pathlib import Path

# Hot Files to optimize
HOT_FILES = [
    "reasoning/traversal.py",
    "core/attention_engine.py",
    "core/reasoning_logit.py"
]

BENCHMARK_CMD = [sys.executable, "benchmarks/ikgwq_metaqa.py", "--sample", "50", "--levels", "0", "2"]

def run_benchmark():
    print("Running benchmark...")
    t0 = time.time()
    try:
        result = subprocess.run(BENCHMARK_CMD, capture_output=True, text=True, check=True)
        elapsed = time.time() - t0
        output = result.stdout
        
        # Parse Hits@10 from results
        # Level 0: H@10=0.3600 (16.6ms/q)
        h10_matches = []
        import re
        matches = re.findall(r"H@10=([\d\.]+)", output)
        if matches:
            h10 = sum(float(m) for m in matches) / len(matches)
        else:
            h10 = 0.0
            
        latency_matches = re.findall(r"\(([\d\.]+)ms/q\)", output)
        if latency_matches:
            latency = sum(float(m) for m in latency_matches) / len(latency_matches)
        else:
            latency = 999.0
            
        return h10, latency, output
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return 0.0, 999.0, str(e)

def main():
    print("=== CEREBRUM Meta-Researcher Baseline ===")
    h10, lat, out = run_benchmark()
    print(f"Baseline H@10: {h10:.4f} | Latency: {lat:.1f}ms/q")
    
    with open("meta_research_log.txt", "a") as f:
        f.write(f"{time.ctime()} - BASELINE - H@10: {h10:.4f}, Latency: {lat:.1f}ms/q\n")

if __name__ == "__main__":
    main()

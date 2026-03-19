"""
Release Validation Script for Parallax.

This script performs end-to-end verification of the primary user journeys:
1. CLI Reasoning Query
2. CLI Community Inspection
3. API Server Lifecycle (Start -> Health Check -> Query -> Stop)

Usage:
    python tests/release_validation.py
"""
import subprocess
import time
import sys
import os
import requests
from pathlib import Path

def run_command(cmd, description):
    print(f"\n>>> Validating: {description}")
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("Result: PASSED")
        return result.stdout
    else:
        print("Result: FAILED")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        sys.exit(1)

def main():
    root = Path(__file__).parent.parent
    csv_path = root / "tests" / "fixtures" / "toy_graph.csv"
    
    # 1. CLI Multi-hop Query
    run_command([
        sys.executable, "-m", "cli.parallax", "query", 
        "--csv", str(csv_path), "newton", "--top-k", "3"
    ], "CLI Reasoning Query")

    # 2. CLI Community Inspection
    run_command([
        sys.executable, "-m", "cli.parallax", "communities", 
        "--csv", str(csv_path)
    ], "CLI Community Inspection")

    # 3. API Server Lifecycle
    print("\n>>> Validating: API Server Lifecycle")
    port = 8200
    server_process = subprocess.Popen([
        sys.executable, "-m", "cli.parallax", "serve",
        "--csv", str(csv_path), "--port", str(port)
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    print(f"Server started on port {port}, waiting for initialization...")
    time.sleep(5)  # Give it time to load graph and detect communities
    
    try:
        # Health Check
        print("Checking /health endpoint...")
        resp = requests.get(f"http://localhost:{port}/health")
        if resp.status_code == 200:
            print(f"Health Check: PASSED (data={resp.json()})")
        else:
            print(f"Health Check: FAILED (status={resp.status_code})")
            sys.exit(1)
            
        # API Query
        print("Checking /query endpoint...")
        query_payload = {"query": "newton", "top_k": 3}
        resp = requests.post(f"http://localhost:{port}/query", json=query_payload)
        if resp.status_code == 200:
            data = resp.json()
            paths = data.get("paths", [])
            print(f"API Query: PASSED (found {len(paths)} paths)")
        else:
            print(f"API Query: FAILED (status={resp.status_code})")
            sys.exit(1)
            
    finally:
        print("Shutting down API server...")
        server_process.terminate()
        server_process.wait()
        print("API Server: Stopped")

    print("\n" + "="*40)
    print("ALL RELEASE JOURNEYS VALIDATED SUCCESSFULLY")
    print("="*40)

if __name__ == "__main__":
    main()

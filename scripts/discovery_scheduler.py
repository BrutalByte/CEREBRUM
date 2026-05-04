"""
Discovery Scheduler: Orchestrates autonomous discovery and synthesis.
Run this using system cron or Windows Task Scheduler at off-peak hours.
"""
import subprocess
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, filename="logs/discovery_scheduler.log")

def run_pipeline():
    logging.info(f"Starting discovery cycle at {datetime.now()}")
    
    # 1. Run Autonomous Discovery
    try:
        subprocess.run(["python", "core/autonomous_researcher.py", "--cycles", "50"], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Discovery failed: {e}")
        return

    # 2. Synthesize Verification Report
    try:
        subprocess.run(["python", "scripts/synthesize_discovery_report.py"], check=True)
        logging.info("Synthesis successful.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Synthesis failed: {e}")

if __name__ == "__main__":
    run_pipeline()

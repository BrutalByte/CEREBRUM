import time
import requests
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cerebrum.monitor")

URL = "http://localhost:8200/v1/health"

def monitor():
    logger.info("Monitor started. Probing %s", URL)
    while True:
        try:
            resp = requests.get(URL, timeout=5)
            if resp.status_code != 200:
                logger.warning("Health check failed (status=%d). Restarting...", resp.status_code)
                subprocess.run(["python", "restart_server.py"])
        except Exception:
            logger.warning("Health check connection refused. Restarting...")
            subprocess.run(["python", "restart_server.py"])
        
        time.sleep(10)

if __name__ == "__main__":
    monitor()


import os
import subprocess
import time
import platform
import signal

def find_and_kill_server():
    print("Searching for existing backend processes...")
    if platform.system() == "Windows":
        # Windows command to find and kill python/uvicorn
        cmds = ["taskkill /f /im uvicorn.exe /t", "taskkill /f /im python.exe /t"]
        for cmd in cmds:
            subprocess.run(cmd, shell=True, capture_output=True)
    else:
        # Unix/Linux command to find and kill
        try:
            subprocess.run("pkill -f 'cli.cerebrum'", shell=True)
        except Exception:
            pass

def restart():
    find_and_kill_server()
    print("Restarting CEREBRUM backend...")
    
    # Adjust this command if your startup parameters differ
    cmd = [
        "python", "-m", "cli.cerebrum", "serve", 
        "--csv", "tests/fixtures/metaqa_movies.csv", 
        "--port", "8200", 
        "--ws-port", "8765"
    ]
    
    try:
        # Using Popen to run in background
        proc = subprocess.Popen(cmd)
        print(f"Backend started with PID: {proc.pid}")
    except Exception as e:
        print(f"Failed to start backend: {e}")

if __name__ == "__main__":
    restart()

import os
import subprocess
import sys

def install_all():
    print("=== CEREBRUM Phase 63+ Telemetry Installation ===")
    
    # 1. Ensure core dependencies
    packages = ["websockets", "pydantic", "fastapi", "uvicorn", "numpy", "networkx"]
    print(f"Installing/Updating dependencies: {', '.join(packages)}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade"] + packages)
    
    # 2. Verify directories
    required_dirs = ["core", "reasoning", "api", "scripts"]
    for d in required_dirs:
        if not os.path.exists(os.path.join("E:\\Development\\Cerebrum", d)):
            print(f"Error: Directory {d} not found.")
            return

    print("\n=== Installation Complete ===")
    print("To launch, run: python scripts/start_cerebrum.py")
    print("Unreal Engine UE5 project is ready to connect via ws://localhost:8765")

if __name__ == "__main__":
    install_all()

import json
import os
from pathlib import Path
from typing import List

HISTORY_FILE = Path("data/cerebrum/history.json")

class StudioHistory:
    """Manages a historical list of recently used databases (graph paths)."""
    
    @staticmethod
    def get_history() -> List[str]:
        """Load history from file."""
        if not HISTORY_FILE.exists():
            return []
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("recent_paths", [])
        except (json.JSONDecodeError, IOError):
            return []

    @staticmethod
    def add_to_history(path: str) -> List[str]:
        """Add a path to history, maintain top 10 unique entries."""
        if not path or not isinstance(path, str):
            return StudioHistory.get_history()
            
        history = StudioHistory.get_history()
        
        # Remove if exists to move to top
        if path in history:
            history.remove(path)
            
        history.insert(0, path)
        history = history[:10]  # Keep last 10
        
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump({"recent_paths": history}, f, indent=4)
            
        return history

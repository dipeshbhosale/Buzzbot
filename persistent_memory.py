import json
import os
from typing import List, Dict, Any
from datetime import datetime

class PersistentMemory:
    def __init__(self, storage_path: str = "conversation_history.json"):
        self.storage_path = storage_path
        self.memory: List[Dict[str, Any]] = self._load_memory()

    def _load_memory(self) -> List[Dict[str, Any]]:
        """Load conversation history from storage."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def add_interaction(self, role: str, content: str) -> None:
        """Add a new interaction to memory."""
        self.memory.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self._save_memory()

    def _save_memory(self) -> None:
        """Save conversation history to storage."""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.memory, f, indent=2)
        except Exception as e:
            print(f"Error saving memory: {e}")

    def get_recent_context(self, n_messages: int = 10) -> str:
        """Get recent conversation history as context."""
        recent = self.memory[-n_messages:] if self.memory else []
        context = []
        for msg in recent:
            prefix = "User" if msg["role"] == "user" else "Assistant"
            context.append(f"{prefix}: {msg['content']}")
        return "\n".join(context)

"""File-based storage implementation."""

import json
import os

from core.interfaces import Storage


class FileStorage(Storage):
    """File-based storage implementation."""

    def __init__(self, config_file: str = None, state_dir: str = None):
        self.config_file = config_file or os.path.expanduser('~/.config/tongue/config.json')
        # Project root is one level up from server/
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.state_dir = state_dir or project_root

    def _get_state_file(self, user_id: str) -> str:
        """Get state file path for a user."""
        if user_id == "default":
            return os.path.join(self.state_dir, 'tongue_state.json')
        return os.path.join(self.state_dir, f'tongue_state_{user_id}.json')

    def load_config(self) -> dict:
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(
                f"Config file not found at {self.config_file}\n"
                f'Please create it with: {{"gemini_api_key": "YOUR_API_KEY_HERE"}}'
            )
        with open(self.config_file, 'r') as f:
            return json.load(f)

    def load_state(self, user_id: str = "default") -> dict | None:
        state_file = self._get_state_file(user_id)
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def save_state(self, state: dict, user_id: str = "default") -> None:
        state_file = self._get_state_file(user_id)
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def list_users(self) -> list[str]:
        """List all existing user IDs."""
        users = []
        if os.path.exists(self.state_dir):
            for filename in os.listdir(self.state_dir):
                if filename == 'tongue_state.json':
                    users.append('default')
                elif filename.startswith('tongue_state_') and filename.endswith('.json'):
                    user_id = filename[13:-5]  # Remove 'tongue_state_' and '.json'
                    users.append(user_id)
        return users

    def user_exists(self, user_id: str) -> bool:
        """Check if a user exists."""
        return os.path.exists(self._get_state_file(user_id))

    def delete_user(self, user_id: str) -> bool:
        """Delete a user's state file."""
        state_file = self._get_state_file(user_id)
        if os.path.exists(state_file):
            os.remove(state_file)
            return True
        return False

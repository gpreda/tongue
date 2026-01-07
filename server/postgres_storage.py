"""PostgreSQL storage implementation."""

import json
import os
import psycopg2
from psycopg2.extras import RealDictCursor

from core.interfaces import Storage


class PostgresStorage(Storage):
    """PostgreSQL-based storage implementation."""

    def __init__(self, config_file: str = None, db_url: str = None):
        self.config_file = config_file or os.path.expanduser('~/.config/tongue/config.json')
        self.db_url = db_url or os.environ.get(
            'DATABASE_URL',
            'postgresql://predator@localhost:5432/tongue'
        )
        self._conn = None

    @property
    def conn(self):
        """Lazy connection initialization."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.db_url)
        return self._conn

    def close(self):
        """Close the database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()

    def load_config(self) -> dict:
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(
                f"Config file not found at {self.config_file}\n"
                f'Please create it with: {{"gemini_api_key": "YOUR_API_KEY_HERE"}}'
            )
        with open(self.config_file, 'r') as f:
            return json.load(f)

    def load_state(self, user_id: str = "default") -> dict | None:
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT state FROM user_state WHERE user_id = %s",
                    (user_id,)
                )
                row = cur.fetchone()
                if row:
                    return row['state']
                return None
        except Exception as e:
            print(f"Error loading state: {e}")
            return None

    def save_state(self, state: dict, user_id: str = "default") -> None:
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO user_state (user_id, state, updated_at)
                    VALUES (%s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (user_id)
                    DO UPDATE SET state = EXCLUDED.state, updated_at = CURRENT_TIMESTAMP
                """, (user_id, json.dumps(state)))
            self.conn.commit()
        except Exception as e:
            print(f"Error saving state: {e}")
            self.conn.rollback()
            raise

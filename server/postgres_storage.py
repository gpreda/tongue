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
        self._initialized = False

    @property
    def conn(self):
        """Lazy connection initialization."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.db_url)
            if not self._initialized:
                self._init_db()
                self._initialized = True
        return self._conn

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_state (
                    user_id VARCHAR(255) PRIMARY KEY,
                    state JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_state_updated
                ON user_state(updated_at)
            """)
        self._conn.commit()

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

    def list_users(self) -> list[str]:
        """List all existing user IDs."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT user_id FROM user_state ORDER BY user_id")
                rows = cur.fetchall()
                return [row[0] for row in rows]
        except Exception as e:
            print(f"Error listing users: {e}")
            return []

    def user_exists(self, user_id: str) -> bool:
        """Check if a user exists."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM user_state WHERE user_id = %s",
                    (user_id,)
                )
                return cur.fetchone() is not None
        except Exception as e:
            print(f"Error checking user: {e}")
            return False

    def delete_user(self, user_id: str) -> bool:
        """Delete a user's state."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM user_state WHERE user_id = %s",
                    (user_id,)
                )
                deleted = cur.rowcount > 0
            self.conn.commit()
            return deleted
        except Exception as e:
            print(f"Error deleting user: {e}")
            self.conn.rollback()
            return False

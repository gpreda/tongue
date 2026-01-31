"""PostgreSQL storage implementation."""

import hashlib
import json
import os
import psycopg2
from psycopg2.extras import RealDictCursor

from core.interfaces import Storage


def hash_pin(pin: str) -> str:
    """Hash a PIN using SHA256."""
    return hashlib.sha256(pin.encode()).hexdigest()


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
        """Lazy connection initialization with stale connection recovery."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.db_url)
            if not self._initialized:
                self._init_db()
                self._initialized = True
        return self._conn

    def _reset_conn(self):
        """Close stale connection so next .conn access reconnects."""
        try:
            if self._conn and not self._conn.closed:
                self._conn.close()
        except Exception:
            pass
        self._conn = None

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_state (
                    user_id VARCHAR(255) PRIMARY KEY,
                    state JSONB NOT NULL,
                    pin_hash VARCHAR(64),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_state_updated
                ON user_state(updated_at)
            """)
            # Add pin_hash column if it doesn't exist (for existing databases)
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'user_state' AND column_name = 'pin_hash'
                    ) THEN
                        ALTER TABLE user_state ADD COLUMN pin_hash VARCHAR(64);
                    END IF;
                END $$;
            """)
            # Languages table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS languages (
                    code VARCHAR(10) PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    script VARCHAR(20) NOT NULL DEFAULT 'latin',
                    english_name VARCHAR(100) NOT NULL,
                    tenses JSONB,
                    accent_words JSONB,
                    active BOOLEAN NOT NULL DEFAULT TRUE
                )
            """)
            # Seed default languages
            cur.execute("""
                INSERT INTO languages (code, name, script, english_name, tenses, accent_words, active)
                VALUES
                    ('es', 'Español', 'latin', 'Spanish',
                     '["present", "preterite", "imperfect", "future", "conditional", "subjunctive"]'::jsonb,
                     '["el", "tu", "mi", "si", "se", "de", "te", "mas", "que", "como", "donde", "cuando", "cual", "quien", "aun", "solo"]'::jsonb,
                     TRUE),
                    ('sr-latn', 'Srpski (latinica)', 'latin', 'Serbian',
                     '["present", "past", "future", "imperative", "conditional"]'::jsonb,
                     '[]'::jsonb,
                     TRUE),
                    ('sr-cyrl', 'Српски (ћирилица)', 'cyrillic', 'Serbian',
                     '["present", "past", "future", "imperative", "conditional"]'::jsonb,
                     '[]'::jsonb,
                     TRUE)
                ON CONFLICT (code) DO NOTHING
            """)
            # Word translations table (shared across all users)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS word_translations (
                    word VARCHAR(255) NOT NULL,
                    language VARCHAR(10) NOT NULL DEFAULT 'es',
                    translation VARCHAR(500) NOT NULL,
                    word_type VARCHAR(50) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (word, language)
                )
            """)
            # Add language column if it doesn't exist (for existing databases)
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'word_translations' AND column_name = 'language'
                    ) THEN
                        ALTER TABLE word_translations ADD COLUMN language VARCHAR(10) NOT NULL DEFAULT 'es';
                        ALTER TABLE word_translations DROP CONSTRAINT IF EXISTS word_translations_pkey;
                        ALTER TABLE word_translations ADD PRIMARY KEY (word, language);
                    END IF;
                END $$;
            """)
            # Verb conjugations table (shared across all users)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS verb_conjugations (
                    conjugated_form VARCHAR(255) NOT NULL,
                    language VARCHAR(10) NOT NULL DEFAULT 'es',
                    base_verb VARCHAR(255) NOT NULL,
                    tense VARCHAR(50) NOT NULL,
                    translation VARCHAR(500) NOT NULL,
                    person VARCHAR(50) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (conjugated_form, language)
                )
            """)
            # Add language column if it doesn't exist (for existing databases)
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'verb_conjugations' AND column_name = 'language'
                    ) THEN
                        ALTER TABLE verb_conjugations ADD COLUMN language VARCHAR(10) NOT NULL DEFAULT 'es';
                        ALTER TABLE verb_conjugations DROP CONSTRAINT IF EXISTS verb_conjugations_pkey;
                        ALTER TABLE verb_conjugations ADD PRIMARY KEY (conjugated_form, language);
                    END IF;
                END $$;
            """)
            # Events log table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    app_name VARCHAR(50) NOT NULL DEFAULT 'tongue',
                    event VARCHAR(50) NOT NULL,
                    user_id VARCHAR(255) NOT NULL,
                    session_id VARCHAR(64),
                    difficulty INTEGER,
                    ms INTEGER,
                    ai_used BOOLEAN DEFAULT FALSE,
                    model_name VARCHAR(100),
                    model_tokens INTEGER,
                    model_ms INTEGER,
                    data JSONB
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_user_id ON events(user_id)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_event ON events(event)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id)
            """)
            # Add app_name column if it doesn't exist (for existing databases)
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'events' AND column_name = 'app_name'
                    ) THEN
                        ALTER TABLE events ADD COLUMN app_name VARCHAR(50) NOT NULL DEFAULT 'tongue';
                    END IF;
                END $$;
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_app_name ON events(app_name)
            """)
            # Add ms column if it doesn't exist (for existing databases)
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'events' AND column_name = 'ms'
                    ) THEN
                        ALTER TABLE events ADD COLUMN ms INTEGER;
                    END IF;
                END $$;
            """)
            # Add AI tracking columns if they don't exist (for existing databases)
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'events' AND column_name = 'ai_used'
                    ) THEN
                        ALTER TABLE events ADD COLUMN ai_used BOOLEAN DEFAULT FALSE;
                    END IF;
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'events' AND column_name = 'model_name'
                    ) THEN
                        ALTER TABLE events ADD COLUMN model_name VARCHAR(100);
                    END IF;
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'events' AND column_name = 'model_tokens'
                    ) THEN
                        ALTER TABLE events ADD COLUMN model_tokens INTEGER;
                    END IF;
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'events' AND column_name = 'model_ms'
                    ) THEN
                        ALTER TABLE events ADD COLUMN model_ms INTEGER;
                    END IF;
                END $$;
            """)
            # API stats table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS api_stats (
                    provider_name VARCHAR(100) PRIMARY KEY,
                    stats JSONB NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Vocabulary items table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS vocabulary_items (
                    category     VARCHAR(50)  NOT NULL,
                    english      VARCHAR(100) NOT NULL,
                    word         VARCHAR(100) NOT NULL,
                    language     VARCHAR(10)  NOT NULL DEFAULT 'es',
                    alternatives VARCHAR(500),
                    PRIMARY KEY (category, english, language)
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_vocab_word
                ON vocabulary_items (word, language)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_vocab_category
                ON vocabulary_items (category, language)
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
        for attempt in range(2):
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
            except psycopg2.OperationalError as e:
                print(f"DB connection error loading state (attempt {attempt + 1}): {e}")
                self._reset_conn()
                if attempt == 1:
                    raise
            except Exception as e:
                print(f"Error loading state: {e}")
                raise

    def save_state(self, state: dict, user_id: str = "default") -> None:
        for attempt in range(2):
            try:
                with self.conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO user_state (user_id, state, updated_at)
                        VALUES (%s, %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (user_id)
                        DO UPDATE SET state = EXCLUDED.state, updated_at = CURRENT_TIMESTAMP
                    """, (user_id, json.dumps(state)))
                self.conn.commit()
                return
            except psycopg2.OperationalError as e:
                print(f"DB connection error saving state (attempt {attempt + 1}): {e}")
                self._reset_conn()
                if attempt == 1:
                    raise
            except Exception as e:
                print(f"Error saving state: {e}")
                try:
                    self.conn.rollback()
                except Exception:
                    self._reset_conn()
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

    def save_pin(self, user_id: str, pin: str) -> bool:
        """Save a hashed PIN for a user."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "UPDATE user_state SET pin_hash = %s WHERE user_id = %s",
                    (hash_pin(pin), user_id)
                )
                updated = cur.rowcount > 0
            self.conn.commit()
            return updated
        except Exception as e:
            print(f"Error saving PIN: {e}")
            self.conn.rollback()
            return False

    def verify_pin(self, user_id: str, pin: str) -> bool:
        """Verify a PIN for a user."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT pin_hash FROM user_state WHERE user_id = %s",
                    (user_id,)
                )
                row = cur.fetchone()
                if row and row[0]:
                    return row[0] == hash_pin(pin)
                return False
        except Exception as e:
            print(f"Error verifying PIN: {e}")
            return False

    def get_pin_hash(self, user_id: str) -> str | None:
        """Get the PIN hash for a user (to check if PIN is set)."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT pin_hash FROM user_state WHERE user_id = %s",
                    (user_id,)
                )
                row = cur.fetchone()
                return row[0] if row else None
        except Exception as e:
            print(f"Error getting PIN hash: {e}")
            return None

    def get_word_translation(self, word: str, language: str = 'es') -> dict | None:
        """Get stored translation for a word."""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT translation, word_type FROM word_translations WHERE word = %s AND language = %s",
                    (word, language)
                )
                row = cur.fetchone()
                if row:
                    return {
                        'translation': row['translation'],
                        'type': row['word_type']
                    }
                return None
        except Exception as e:
            print(f"Error getting word translation: {e}")
            return None

    def save_word_translation(self, word: str, translation: str, word_type: str, language: str = 'es') -> None:
        """Save translation for a word."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO word_translations (word, language, translation, word_type)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (word, language) DO UPDATE SET
                        translation = EXCLUDED.translation,
                        word_type = EXCLUDED.word_type
                """, (word, language, translation, word_type))
            self.conn.commit()
        except Exception as e:
            print(f"Error saving word translation: {e}")
            self.conn.rollback()
            raise

    def get_verb_conjugation(self, conjugated_form: str, language: str = 'es') -> dict | None:
        """Get stored conjugation info for a verb form."""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """SELECT base_verb, tense, translation, person
                       FROM verb_conjugations WHERE conjugated_form = %s AND language = %s""",
                    (conjugated_form, language)
                )
                row = cur.fetchone()
                if row:
                    return {
                        'base_verb': row['base_verb'],
                        'tense': row['tense'],
                        'translation': row['translation'],
                        'person': row['person']
                    }
                return None
        except Exception as e:
            print(f"Error getting verb conjugation: {e}")
            return None

    def save_verb_conjugation(self, conjugated_form: str, base_verb: str, tense: str,
                              translation: str, person: str, language: str = 'es') -> None:
        """Save conjugation info for a verb form."""
        # Use 'n/a' for infinitives and other forms without grammatical person
        if person is None:
            person = 'n/a'
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO verb_conjugations (conjugated_form, language, base_verb, tense, translation, person)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (conjugated_form, language) DO UPDATE SET
                        base_verb = EXCLUDED.base_verb,
                        tense = EXCLUDED.tense,
                        translation = EXCLUDED.translation,
                        person = EXCLUDED.person
                """, (conjugated_form, language, base_verb, tense, translation, person))
            self.conn.commit()
        except Exception as e:
            print(f"Error saving verb conjugation: {e}")
            self.conn.rollback()
            raise

    # Event logging methods
    def log_event(self, event: str, user_id: str, session_id: str = None,
                  difficulty: int = None, app_name: str = "tongue", ms: int = None,
                  ai_used: bool = False, model_name: str = None,
                  model_tokens: int = None, model_ms: int = None, **data) -> None:
        """Log an event to the database."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO events (app_name, event, user_id, session_id, difficulty, ms,
                                        ai_used, model_name, model_tokens, model_ms, data)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (app_name, event, user_id, session_id, difficulty, ms,
                      ai_used, model_name, model_tokens, model_ms,
                      json.dumps(data) if data else None))
            self.conn.commit()
        except Exception as e:
            print(f"Error logging event: {e}")
            self.conn.rollback()

    def get_user_events(self, user_id: str, event_type: str = None,
                        limit: int = 100) -> list[dict]:
        """Get recent events for a user."""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                if event_type:
                    cur.execute("""
                        SELECT * FROM events
                        WHERE user_id = %s AND event = %s
                        ORDER BY timestamp DESC LIMIT %s
                    """, (user_id, event_type, limit))
                else:
                    cur.execute("""
                        SELECT * FROM events
                        WHERE user_id = %s
                        ORDER BY timestamp DESC LIMIT %s
                    """, (user_id, limit))
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            print(f"Error getting user events: {e}")
            return []

    def get_events(self, user_id: str = None, event_type: str = None,
                   app_name: str = None, limit: int = 100) -> list[dict]:
        """Get recent events with optional filters."""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                conditions = []
                params = []
                if user_id:
                    conditions.append("user_id = %s")
                    params.append(user_id)
                if event_type:
                    conditions.append("event = %s")
                    params.append(event_type)
                if app_name:
                    conditions.append("app_name = %s")
                    params.append(app_name)

                where_clause = " AND ".join(conditions) if conditions else "1=1"
                params.append(limit)

                cur.execute(f"""
                    SELECT * FROM events
                    WHERE {where_clause}
                    ORDER BY timestamp DESC LIMIT %s
                """, params)
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            print(f"Error getting events: {e}")
            return []

    def get_user_stats(self, user_id: str) -> dict:
        """Get aggregated stats for a user."""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Total translations and average score
                cur.execute("""
                    SELECT
                        COUNT(*) as total_translations,
                        AVG((data->>'score')::float) as avg_score,
                        COUNT(*) FILTER (WHERE (data->>'score')::int >= 80) as good_translations,
                        COUNT(*) FILTER (WHERE (data->>'score')::int < 50) as poor_translations
                    FROM events
                    WHERE user_id = %s AND event = 'translation.result'
                """, (user_id,))
                translation_stats = dict(cur.fetchone())

                # Hints used
                cur.execute("""
                    SELECT COUNT(*) as hints_used
                    FROM events
                    WHERE user_id = %s AND event = 'hint.request'
                """, (user_id,))
                hint_stats = dict(cur.fetchone())

                # Stories generated
                cur.execute("""
                    SELECT
                        COUNT(*) as stories_generated,
                        COUNT(*) FILTER (WHERE (data->>'from_cache')::boolean = true) as from_cache
                    FROM events
                    WHERE user_id = %s AND event = 'story.generate'
                """, (user_id,))
                story_stats = dict(cur.fetchone())

                # Level changes
                cur.execute("""
                    SELECT
                        COUNT(*) FILTER (WHERE data->>'direction' = 'up') as level_ups,
                        COUNT(*) FILTER (WHERE data->>'direction' = 'down') as level_downs
                    FROM events
                    WHERE user_id = %s AND event = 'level.change'
                """, (user_id,))
                level_stats = dict(cur.fetchone())

                # Sessions count
                cur.execute("""
                    SELECT COUNT(DISTINCT session_id) as sessions
                    FROM events
                    WHERE user_id = %s AND session_id IS NOT NULL
                """, (user_id,))
                session_stats = dict(cur.fetchone())

                # Challenge stats
                cur.execute("""
                    SELECT
                        data->>'challenge_type' as challenge_type,
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE (data->>'score')::int >= 100) as correct
                    FROM events
                    WHERE user_id = %s AND event = 'translation.result'
                        AND data->>'challenge_type' IS NOT NULL
                    GROUP BY data->>'challenge_type'
                """, (user_id,))
                challenge_rows = cur.fetchall()
                challenge_stats = {row['challenge_type']: {'total': row['total'], 'correct': row['correct']}
                                   for row in challenge_rows}

                return {
                    **translation_stats,
                    **hint_stats,
                    **story_stats,
                    **level_stats,
                    **session_stats,
                    'challenge_stats': challenge_stats
                }
        except Exception as e:
            print(f"Error getting user stats: {e}")
            return {}

    def get_global_stats(self) -> dict:
        """Get aggregated stats across all users."""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        COUNT(DISTINCT user_id) as total_users,
                        COUNT(*) FILTER (WHERE event = 'translation.result') as total_translations,
                        COUNT(*) FILTER (WHERE event = 'story.generate') as total_stories,
                        COUNT(*) FILTER (WHERE event = 'hint.request') as total_hints,
                        COUNT(DISTINCT session_id) as total_sessions
                    FROM events
                """)
                return dict(cur.fetchone())
        except Exception as e:
            print(f"Error getting global stats: {e}")
            return {}

    def get_app_names(self) -> list[str]:
        """Get all unique app names from events."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT app_name FROM events ORDER BY app_name
                """)
                return [row[0] for row in cur.fetchall()]
        except Exception as e:
            print(f"Error getting app names: {e}")
            return []

    def get_event_users(self) -> list[str]:
        """Get all unique user IDs from events."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT user_id FROM events ORDER BY user_id
                """)
                return [row[0] for row in cur.fetchall()]
        except Exception as e:
            print(f"Error getting event users: {e}")
            return []

    def get_perf_stats(self, app_name: str = None) -> list[dict]:
        """Get performance statistics grouped by event type."""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                app_filter = "AND app_name = %s" if app_name else ""
                params = (app_name,) if app_name else ()

                cur.execute(f"""
                    WITH event_stats AS (
                        SELECT
                            event,
                            COUNT(*) as invocations,
                            COUNT(DISTINCT session_id) as sessions,
                            COUNT(*) FILTER (WHERE ms IS NOT NULL) as ms_count,
                            AVG(ms) FILTER (WHERE ms IS NOT NULL) as avg_ms,
                            MIN(ms) FILTER (WHERE ms IS NOT NULL) as min_ms,
                            MAX(ms) FILTER (WHERE ms IS NOT NULL) as max_ms,
                            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ms) FILTER (WHERE ms IS NOT NULL) as p50_ms,
                            PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY ms) FILTER (WHERE ms IS NOT NULL) as p90_ms,
                            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ms) FILTER (WHERE ms IS NOT NULL) as p95_ms,
                            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY ms) FILTER (WHERE ms IS NOT NULL) as p99_ms,
                            COUNT(*) FILTER (WHERE ai_used = TRUE) as ai_invocations,
                            AVG(model_ms) FILTER (WHERE model_ms IS NOT NULL) as avg_model_ms,
                            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY model_ms) FILTER (WHERE model_ms IS NOT NULL) as p50_model_ms,
                            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY model_ms) FILTER (WHERE model_ms IS NOT NULL) as p95_model_ms,
                            SUM(model_tokens) FILTER (WHERE model_tokens IS NOT NULL) as total_tokens,
                            AVG(model_tokens) FILTER (WHERE model_tokens IS NOT NULL) as avg_tokens
                        FROM events
                        WHERE 1=1 {app_filter}
                        GROUP BY event
                    )
                    SELECT
                        event,
                        invocations,
                        sessions,
                        CASE WHEN sessions > 0 THEN ROUND(invocations::numeric / sessions, 2) ELSE 0 END as avg_per_session,
                        ms_count,
                        ROUND(avg_ms::numeric, 1) as avg_ms,
                        min_ms,
                        max_ms,
                        ROUND(p50_ms::numeric, 1) as p50_ms,
                        ROUND(p90_ms::numeric, 1) as p90_ms,
                        ROUND(p95_ms::numeric, 1) as p95_ms,
                        ROUND(p99_ms::numeric, 1) as p99_ms,
                        ai_invocations,
                        ROUND(avg_model_ms::numeric, 1) as avg_model_ms,
                        ROUND(p50_model_ms::numeric, 1) as p50_model_ms,
                        ROUND(p95_model_ms::numeric, 1) as p95_model_ms,
                        total_tokens,
                        ROUND(avg_tokens::numeric, 0) as avg_tokens
                    FROM event_stats
                    ORDER BY invocations DESC
                """, params)
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            print(f"Error getting perf stats: {e}")
            return []

    def load_api_stats(self, provider_name: str) -> dict | None:
        """Load API usage stats for a provider."""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT stats FROM api_stats WHERE provider_name = %s",
                    (provider_name,)
                )
                row = cur.fetchone()
                if row:
                    return row['stats']
                return None
        except Exception as e:
            print(f"Error loading API stats: {e}")
            return None

    def save_api_stats(self, provider_name: str, stats: dict) -> None:
        """Save API usage stats for a provider."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO api_stats (provider_name, stats, updated_at)
                    VALUES (%s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (provider_name)
                    DO UPDATE SET stats = EXCLUDED.stats, updated_at = CURRENT_TIMESTAMP
                """, (provider_name, json.dumps(stats)))
            self.conn.commit()
        except Exception as e:
            print(f"Error saving API stats: {e}")
            self.conn.rollback()

    # Vocabulary storage methods

    def seed_vocabulary(self, items: list[dict]) -> None:
        """Seed vocabulary items into the database if not already seeded for this language."""
        if not items:
            return
        language = items[0].get('language', 'es')
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM vocabulary_items WHERE language = %s", (language,))
                count = cur.fetchone()[0]
                if count > 0:
                    return  # Already seeded for this language

                for item in items:
                    cur.execute("""
                        INSERT INTO vocabulary_items (category, english, word, language, alternatives)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (category, english, language) DO NOTHING
                    """, (item['category'], item['english'], item['word'],
                          item['language'], item['alternatives']))
            self.conn.commit()
            print(f"Seeded {len(items)} vocabulary items for language '{language}'")
        except Exception as e:
            print(f"Error seeding vocabulary: {e}")
            self.conn.rollback()

    def get_vocab_categories(self, language: str = 'es') -> list[str]:
        """Get distinct vocabulary categories."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT category FROM vocabulary_items WHERE language = %s ORDER BY category",
                    (language,)
                )
                return [row[0] for row in cur.fetchall()]
        except Exception as e:
            print(f"Error getting vocab categories: {e}")
            return []

    def get_vocab_category_items(self, category: str, language: str = 'es') -> list[dict]:
        """Get vocabulary items for a category."""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """SELECT english, word, alternatives
                       FROM vocabulary_items
                       WHERE category = %s AND language = %s""",
                    (category, language)
                )
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            print(f"Error getting vocab category items: {e}")
            return []

    def get_vocab_item_by_english(self, category: str, english: str, language: str = 'es') -> dict | None:
        """Look up a single vocabulary item by category and english key."""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """SELECT english, word, alternatives
                       FROM vocabulary_items
                       WHERE category = %s AND english = %s AND language = %s""",
                    (category, english, language)
                )
                row = cur.fetchone()
                return dict(row) if row else None
        except Exception as e:
            print(f"Error getting vocab item by english: {e}")
            return None

    def get_languages(self) -> list[dict]:
        """Get all active languages."""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """SELECT code, name, script, english_name, tenses, accent_words
                       FROM languages WHERE active = TRUE ORDER BY code"""
                )
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            print(f"Error getting languages: {e}")
            return []

    def get_language(self, code: str) -> dict | None:
        """Get language info by code."""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """SELECT code, name, script, english_name, tenses, accent_words
                       FROM languages WHERE code = %s""",
                    (code,)
                )
                row = cur.fetchone()
                return dict(row) if row else None
        except Exception as e:
            print(f"Error getting language: {e}")
            return None

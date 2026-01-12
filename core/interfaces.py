"""Abstract base classes for dependency injection."""

from abc import ABC, abstractmethod


class AIProvider(ABC):
    """Abstract base class for AI/LLM provider."""

    @abstractmethod
    def generate_story(self, correct_words: list, difficulty: int) -> tuple[str, int]:
        """Generate a story. Returns (story_text, generation_time_ms)."""
        pass

    @abstractmethod
    def validate_translation(self, sentence: str, translation: str) -> tuple[dict, int]:
        """Validate a translation. Returns (judgement_dict, validation_time_ms)."""
        pass

    @abstractmethod
    def get_hint(self, sentence: str, correct_words: list) -> dict | None:
        """Get a hint for translation. Returns hint dict or None."""
        pass


class Storage(ABC):
    """Abstract base class for state and config storage."""

    @abstractmethod
    def load_config(self) -> dict:
        """Load configuration. Returns config dict."""
        pass

    @abstractmethod
    def load_state(self, user_id: str = "default") -> dict | None:
        """Load state for a user. Returns state dict or None if not found."""
        pass

    @abstractmethod
    def save_state(self, state: dict, user_id: str = "default") -> None:
        """Save state for a user."""
        pass

    @abstractmethod
    def save_pin(self, user_id: str, pin: str) -> bool:
        """Save a hashed PIN for a user. Returns True on success."""
        pass

    @abstractmethod
    def verify_pin(self, user_id: str, pin: str) -> bool:
        """Verify a PIN for a user. Returns True if PIN matches."""
        pass

    @abstractmethod
    def get_pin_hash(self, user_id: str) -> str | None:
        """Get the PIN hash for a user (to check if PIN is set)."""
        pass

    @abstractmethod
    def get_word_translation(self, word: str) -> dict | None:
        """Get stored translation for a word. Returns {translation, type} or None."""
        pass

    @abstractmethod
    def save_word_translation(self, word: str, translation: str, word_type: str) -> None:
        """Save translation for a word."""
        pass

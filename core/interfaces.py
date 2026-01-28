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

    @abstractmethod
    def get_verb_conjugation(self, conjugated_form: str) -> dict | None:
        """Get stored conjugation info for a verb form.
        Returns {base_verb, tense, translation, person} or None."""
        pass

    @abstractmethod
    def save_verb_conjugation(self, conjugated_form: str, base_verb: str, tense: str,
                              translation: str, person: str) -> None:
        """Save conjugation info for a verb form."""
        pass

    @abstractmethod
    def load_api_stats(self, provider_name: str) -> dict | None:
        """Load API usage stats for a provider. Returns stats dict or None."""
        pass

    @abstractmethod
    def save_api_stats(self, provider_name: str, stats: dict) -> None:
        """Save API usage stats for a provider."""
        pass

    @abstractmethod
    def seed_vocabulary(self, items: list[dict]) -> None:
        """Seed vocabulary items into storage. items are dicts with
        {category, english, word, language, alternatives}."""
        pass

    @abstractmethod
    def get_vocab_categories(self, language: str = 'es') -> list[str]:
        """Get distinct vocabulary categories for a language."""
        pass

    @abstractmethod
    def get_vocab_category_items(self, category: str, language: str = 'es') -> list[dict]:
        """Get vocabulary items for a category.
        Returns list of {english, word, alternatives} dicts."""
        pass

    @abstractmethod
    def get_vocab_item_by_english(self, category: str, english: str, language: str = 'es') -> dict | None:
        """Look up a single vocabulary item by category and english key.
        Returns {english, word, alternatives} or None."""
        pass

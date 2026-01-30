"""File-based storage implementation."""

import hashlib
import json
import os

from core.interfaces import Storage


def hash_pin(pin: str) -> str:
    """Hash a PIN using SHA256."""
    return hashlib.sha256(pin.encode()).hexdigest()


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
            with open(state_file, 'r') as f:
                return json.load(f)
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
            # Also remove PIN if stored
            pins = self._load_pins()
            if user_id in pins:
                del pins[user_id]
                self._save_pins(pins)
            return True
        return False

    def _get_pins_file(self) -> str:
        """Get the path to the pins file."""
        return os.path.join(self.state_dir, 'tongue_pins.json')

    def _load_pins(self) -> dict:
        """Load all PIN hashes."""
        pins_file = self._get_pins_file()
        if os.path.exists(pins_file):
            try:
                with open(pins_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_pins(self, pins: dict) -> None:
        """Save all PIN hashes."""
        pins_file = self._get_pins_file()
        with open(pins_file, 'w') as f:
            json.dump(pins, f, indent=2)

    def save_pin(self, user_id: str, pin: str) -> bool:
        """Save a hashed PIN for a user."""
        try:
            pins = self._load_pins()
            pins[user_id] = hash_pin(pin)
            self._save_pins(pins)
            return True
        except Exception as e:
            print(f"Error saving PIN: {e}")
            return False

    def verify_pin(self, user_id: str, pin: str) -> bool:
        """Verify a PIN for a user."""
        try:
            pins = self._load_pins()
            stored_hash = pins.get(user_id)
            if stored_hash:
                return stored_hash == hash_pin(pin)
            return False
        except Exception as e:
            print(f"Error verifying PIN: {e}")
            return False

    def get_pin_hash(self, user_id: str) -> str | None:
        """Get the PIN hash for a user (to check if PIN is set)."""
        try:
            pins = self._load_pins()
            return pins.get(user_id)
        except Exception as e:
            print(f"Error getting PIN hash: {e}")
            return None

    def _get_translations_file(self) -> str:
        """Get the path to the word translations file."""
        return os.path.join(self.state_dir, 'tongue_translations.json')

    def _load_translations(self) -> dict:
        """Load all word translations."""
        trans_file = self._get_translations_file()
        if os.path.exists(trans_file):
            try:
                with open(trans_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_translations(self, translations: dict) -> None:
        """Save all word translations."""
        trans_file = self._get_translations_file()
        with open(trans_file, 'w') as f:
            json.dump(translations, f, indent=2)

    def get_word_translation(self, word: str) -> dict | None:
        """Get stored translation for a word."""
        try:
            translations = self._load_translations()
            return translations.get(word)
        except Exception as e:
            print(f"Error getting word translation: {e}")
            return None

    def save_word_translation(self, word: str, translation: str, word_type: str) -> None:
        """Save translation for a word."""
        try:
            translations = self._load_translations()
            translations[word] = {
                'translation': translation,
                'type': word_type
            }
            self._save_translations(translations)
        except Exception as e:
            print(f"Error saving word translation: {e}")
            raise

    def _get_conjugations_file(self) -> str:
        """Get the path to the verb conjugations file."""
        return os.path.join(self.state_dir, 'tongue_conjugations.json')

    def _load_conjugations(self) -> dict:
        """Load all verb conjugations."""
        conj_file = self._get_conjugations_file()
        if os.path.exists(conj_file):
            try:
                with open(conj_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_conjugations(self, conjugations: dict) -> None:
        """Save all verb conjugations."""
        conj_file = self._get_conjugations_file()
        with open(conj_file, 'w') as f:
            json.dump(conjugations, f, indent=2)

    def get_verb_conjugation(self, conjugated_form: str) -> dict | None:
        """Get stored conjugation info for a verb form."""
        try:
            conjugations = self._load_conjugations()
            return conjugations.get(conjugated_form)
        except Exception as e:
            print(f"Error getting verb conjugation: {e}")
            return None

    def save_verb_conjugation(self, conjugated_form: str, base_verb: str, tense: str,
                              translation: str, person: str) -> None:
        """Save conjugation info for a verb form."""
        # Use 'n/a' for infinitives and other forms without grammatical person
        if person is None:
            person = 'n/a'
        try:
            conjugations = self._load_conjugations()
            conjugations[conjugated_form] = {
                'base_verb': base_verb,
                'tense': tense,
                'translation': translation,
                'person': person
            }
            self._save_conjugations(conjugations)
        except Exception as e:
            print(f"Error saving verb conjugation: {e}")
            raise

    def _get_api_stats_file(self) -> str:
        """Get the path to the API stats file."""
        return os.path.join(self.state_dir, 'tongue_api_stats.json')

    def _load_all_api_stats(self) -> dict:
        """Load all API stats."""
        stats_file = self._get_api_stats_file()
        if os.path.exists(stats_file):
            try:
                with open(stats_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_all_api_stats(self, all_stats: dict) -> None:
        """Save all API stats."""
        stats_file = self._get_api_stats_file()
        with open(stats_file, 'w') as f:
            json.dump(all_stats, f, indent=2)

    def load_api_stats(self, provider_name: str) -> dict | None:
        """Load API usage stats for a provider."""
        try:
            all_stats = self._load_all_api_stats()
            return all_stats.get(provider_name)
        except Exception as e:
            print(f"Error loading API stats: {e}")
            return None

    def save_api_stats(self, provider_name: str, stats: dict) -> None:
        """Save API usage stats for a provider."""
        try:
            all_stats = self._load_all_api_stats()
            all_stats[provider_name] = stats
            self._save_all_api_stats(all_stats)
        except Exception as e:
            print(f"Error saving API stats: {e}")
            raise

    # Vocabulary storage methods (file storage uses static dict, no DB)

    def seed_vocabulary(self, items: list[dict]) -> None:
        """No-op for file storage — vocabulary lives in the static dict."""
        pass

    def get_vocab_categories(self, language: str = 'es') -> list[str]:
        """Get distinct vocabulary categories from static dict."""
        from core.vocabulary import VOCABULARY_CHALLENGES
        return list(VOCABULARY_CHALLENGES.keys())

    def get_vocab_category_items(self, category: str, language: str = 'es') -> list[dict]:
        """Get vocabulary items for a category from static dict."""
        from core.vocabulary import VOCABULARY_CHALLENGES
        data = VOCABULARY_CHALLENGES.get(category, {}).get('items', {})
        result = []
        for word, alternatives in data.items():
            english = alternatives.split(',')[0].strip()
            result.append({
                'english': english,
                'word': word,
                'alternatives': alternatives
            })
        return result

    def get_vocab_item_by_english(self, category: str, english: str, language: str = 'es') -> dict | None:
        """Look up a single vocabulary item by category and english key from static dict."""
        from core.vocabulary import VOCABULARY_CHALLENGES
        data = VOCABULARY_CHALLENGES.get(category, {}).get('items', {})
        for word, alternatives in data.items():
            key = alternatives.split(',')[0].strip()
            if key == english:
                return {
                    'english': english,
                    'word': word,
                    'alternatives': alternatives
                }
        return None

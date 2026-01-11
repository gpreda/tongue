"""Unit tests for tongue core module."""

import unittest
from unittest.mock import MagicMock
import sys

# Mock google.generativeai before importing
sys.modules['google'] = MagicMock()
sys.modules['google.generativeai'] = MagicMock()

from core.models import TongueRound, History
from core.interfaces import AIProvider, Storage
from core.utils import split_into_sentences
from core.config import (
    MIN_DIFFICULTY, MAX_DIFFICULTY,
    ADVANCE_WINDOW_SIZE, ADVANCE_SCORE_THRESHOLD, ADVANCE_REQUIRED_GOOD,
    DEMOTE_SCORE_THRESHOLD, DEMOTE_REQUIRED_POOR
)


# ============================================================================
# Mock Implementations
# ============================================================================

class MockAIProvider(AIProvider):
    """Mock AI provider for testing."""

    def __init__(self):
        self.stories = []
        self.validations = []
        self.hints = []
        self.generate_story_calls = []
        self.validate_translation_calls = []
        self.get_hint_calls = []

    def set_story_response(self, story: str, ms: int = 100):
        """Queue a story response."""
        self.stories.append((story, ms))

    def set_validation_response(self, judgement: dict, ms: int = 50):
        """Queue a validation response."""
        self.validations.append((judgement, ms))

    def set_hint_response(self, hint: dict | None):
        """Queue a hint response."""
        self.hints.append(hint)

    def generate_story(self, correct_words: list, difficulty: int) -> tuple[str, int]:
        self.generate_story_calls.append((correct_words.copy(), difficulty))
        if self.stories:
            return self.stories.pop(0)
        return ("Default story sentence one. Default story sentence two.", 100)

    def validate_translation(self, sentence: str, translation: str) -> tuple[dict, int]:
        self.validate_translation_calls.append((sentence, translation))
        if self.validations:
            return self.validations.pop(0)
        return ({
            'score': 80,
            'correct_translation': 'Default translation',
            'evaluation': 'Good translation',
            'vocabulary_breakdown': []
        }, 50)

    def get_hint(self, sentence: str, correct_words: list) -> dict | None:
        self.get_hint_calls.append((sentence, correct_words.copy()))
        if self.hints:
            return self.hints.pop(0)
        return {'noun': ['casa', 'house'], 'verb': ['comer', 'to eat']}


class MockStorage(Storage):
    """Mock storage for testing."""

    def __init__(self):
        self.config = {'gemini_api_key': 'test-api-key'}
        self.state = None
        self.save_calls = []

    def set_config(self, config: dict):
        self.config = config

    def set_state(self, state: dict | None):
        self.state = state

    def load_config(self) -> dict:
        return self.config

    def load_state(self, user_id: str = "default") -> dict | None:
        return self.state

    def save_state(self, state: dict, user_id: str = "default") -> None:
        self.save_calls.append(state.copy())
        self.state = state


# ============================================================================
# Test Cases
# ============================================================================

class TestSplitIntoSentences(unittest.TestCase):
    """Tests for split_into_sentences utility function."""

    def test_simple_sentences(self):
        text = "Hello world. How are you? I am fine!"
        result = split_into_sentences(text)
        self.assertEqual(result, ["Hello world.", "How are you?", "I am fine!"])

    def test_multiple_spaces(self):
        text = "First sentence.   Second sentence.    Third sentence."
        result = split_into_sentences(text)
        self.assertEqual(result, ["First sentence.", "Second sentence.", "Third sentence."])

    def test_numbered_sentences(self):
        text = "1. First item. 2) Second item. 3. Third item."
        result = split_into_sentences(text)
        self.assertEqual(result, ["First item.", "Second item.", "Third item."])

    def test_filters_short_sentences(self):
        text = "OK. This is a proper sentence. No."
        result = split_into_sentences(text)
        self.assertEqual(result, ["This is a proper sentence."])

    def test_empty_string(self):
        result = split_into_sentences("")
        self.assertEqual(result, [])

    def test_whitespace_handling(self):
        text = "  First sentence.  Second sentence.  "
        result = split_into_sentences(text)
        self.assertEqual(result, ["First sentence.", "Second sentence."])


class TestTongueRound(unittest.TestCase):
    """Tests for TongueRound class."""

    def test_initialization(self):
        round = TongueRound("Hola mundo.", 3, 150)
        self.assertEqual(round.sentence, "Hola mundo.")
        self.assertEqual(round.difficulty, 3)
        self.assertEqual(round.generate_ms, 150)
        self.assertIsNone(round.translation)
        self.assertIsNone(round.judgement)
        self.assertFalse(round.evaluated)

    def test_to_dict(self):
        round = TongueRound("Hola mundo.", 3, 150)
        round.translation = "Hello world"
        round.judgement = {'score': 95}
        round.evaluated = True
        round.judge_ms = 50

        data = round.to_dict()
        self.assertEqual(data['sentence'], "Hola mundo.")
        self.assertEqual(data['difficulty'], 3)
        self.assertEqual(data['translation'], "Hello world")
        self.assertEqual(data['judgement'], {'score': 95})
        self.assertTrue(data['evaluated'])

    def test_from_dict(self):
        data = {
            'sentence': "Hola mundo.",
            'difficulty': 3,
            'generate_ms': 150,
            'translation': "Hello world",
            'judgement': {'score': 95},
            'evaluated': True,
            'judge_ms': 50
        }
        round = TongueRound.from_dict(data)
        self.assertEqual(round.sentence, "Hola mundo.")
        self.assertEqual(round.difficulty, 3)
        self.assertEqual(round.translation, "Hello world")
        self.assertEqual(round.get_score(), 95)

    def test_get_score_with_judgement(self):
        round = TongueRound("Test", 1, 100)
        round.judgement = {'score': 85}
        self.assertEqual(round.get_score(), 85)

    def test_get_score_without_judgement(self):
        round = TongueRound("Test", 1, 100)
        self.assertEqual(round.get_score(), 0)


class TestHistory(unittest.TestCase):
    """Tests for History class."""

    def test_initialization(self):
        history = History()
        self.assertEqual(history.difficulty, MIN_DIFFICULTY)
        self.assertEqual(history.total_completed, 0)
        self.assertEqual(history.correct_words, [])
        self.assertEqual(history.level_scores, [])

    def test_record_score_same_level(self):
        history = History()
        history.difficulty = 3
        history.record_score(3, 85)
        history.record_score(3, 90)
        self.assertEqual(history.level_scores, [85, 90])

    def test_record_score_different_level_ignored(self):
        history = History()
        history.difficulty = 3
        history.record_score(2, 85)  # Different level, should be ignored
        self.assertEqual(history.level_scores, [])

    def test_record_score_window_limit(self):
        history = History()
        history.difficulty = 1
        for i in range(10):
            history.record_score(1, 50 + i)
        self.assertEqual(len(history.level_scores), ADVANCE_WINDOW_SIZE)
        # Should keep the most recent scores
        self.assertEqual(history.level_scores[-1], 59)

    def test_get_good_score_count(self):
        history = History()
        history.level_scores = [90, 85, 70, 95, 60]  # 3 good (>=80)
        self.assertEqual(history.get_good_score_count(), 3)

    def test_get_poor_score_count(self):
        history = History()
        history.level_scores = [40, 45, 70, 30, 60]  # 3 poor (<50)
        self.assertEqual(history.get_poor_score_count(), 3)

    def test_check_advancement_success(self):
        history = History()
        history.difficulty = 3
        history.level_scores = [90, 85, 90, 95, 85]  # 5 good scores
        self.assertTrue(history.check_advancement())

    def test_check_advancement_not_enough_good(self):
        history = History()
        history.difficulty = 3
        history.level_scores = [90, 85, 70, 60, 50]  # Only 2 good
        self.assertFalse(history.check_advancement())

    def test_check_advancement_at_max_level(self):
        history = History()
        history.difficulty = MAX_DIFFICULTY
        history.level_scores = [90, 85, 90, 95, 85]
        self.assertFalse(history.check_advancement())

    def test_check_advancement_not_enough_attempts(self):
        history = History()
        history.difficulty = 3
        history.level_scores = [90, 85, 90]  # Only 3 attempts
        self.assertFalse(history.check_advancement())

    def test_check_demotion_success(self):
        history = History()
        history.difficulty = 5
        history.level_scores = [30, 40, 45, 35, 60]  # 4 poor scores
        self.assertTrue(history.check_demotion())

    def test_check_demotion_not_enough_poor(self):
        history = History()
        history.difficulty = 5
        history.level_scores = [30, 60, 70, 80, 55]  # Only 1 poor
        self.assertFalse(history.check_demotion())

    def test_check_demotion_at_min_level(self):
        history = History()
        history.difficulty = MIN_DIFFICULTY
        history.level_scores = [30, 40, 45, 35, 20]
        self.assertFalse(history.check_demotion())

    def test_advance_level(self):
        history = History()
        history.difficulty = 3
        history.level_scores = [90, 85]
        history.story_sentences = ["sentence"]
        history.current_story = "story"

        result = history.advance_level()

        self.assertTrue(result)
        self.assertEqual(history.difficulty, 4)
        self.assertEqual(history.level_scores, [])
        self.assertEqual(history.story_sentences, [])
        self.assertIsNone(history.current_story)

    def test_advance_level_at_max(self):
        history = History()
        history.difficulty = MAX_DIFFICULTY
        result = history.advance_level()
        self.assertFalse(result)
        self.assertEqual(history.difficulty, MAX_DIFFICULTY)

    def test_demote_level(self):
        history = History()
        history.difficulty = 5
        history.level_scores = [30, 40]

        result = history.demote_level()

        self.assertTrue(result)
        self.assertEqual(history.difficulty, 4)
        self.assertEqual(history.level_scores, [])

    def test_demote_level_at_min(self):
        history = History()
        history.difficulty = MIN_DIFFICULTY
        result = history.demote_level()
        self.assertFalse(result)
        self.assertEqual(history.difficulty, MIN_DIFFICULTY)

    def test_set_story(self):
        history = History()
        story = "First sentence. Second sentence. Third sentence."
        history.set_story(story, 3, 200)

        self.assertEqual(history.current_story, story)
        self.assertEqual(history.story_difficulty, 3)
        self.assertEqual(history.story_generate_ms, 200)
        self.assertEqual(len(history.story_sentences), 3)

    def test_needs_new_story_when_empty(self):
        history = History()
        self.assertTrue(history.needs_new_story())

    def test_needs_new_story_different_difficulty(self):
        history = History()
        history.story_sentences = ["sentence"]
        history.story_difficulty = 3
        history.difficulty = 4
        self.assertTrue(history.needs_new_story())

    def test_needs_new_story_same_difficulty(self):
        history = History()
        history.story_sentences = ["sentence"]
        history.story_difficulty = 3
        history.difficulty = 3
        self.assertFalse(history.needs_new_story())

    def test_get_next_sentence(self):
        history = History()
        history.story_sentences = ["First.", "Second.", "Third."]
        history.story_difficulty = 3
        history.story_generate_ms = 100

        round = history.get_next_sentence()

        self.assertEqual(round.sentence, "First.")
        self.assertEqual(round.difficulty, 3)
        self.assertEqual(len(history.story_sentences), 2)
        self.assertEqual(len(history.rounds), 1)

    def test_get_next_sentence_empty(self):
        history = History()
        history.story_sentences = []
        round = history.get_next_sentence()
        self.assertIsNone(round)

    def test_update_words_correct_noun(self):
        history = History()
        round = TongueRound("Test", 1, 100)
        round.judgement = {
            'vocabulary_breakdown': [
                ['casa', 'house', 'noun', True],
                ['grande', 'big', 'adjective', True]
            ]
        }
        history.update_words(round)
        self.assertIn('casa', history.correct_words)
        self.assertNotIn('grande', history.correct_words)  # Adjectives not tracked

    def test_update_words_missed_word(self):
        history = History()
        round = TongueRound("Test", 1, 100)
        round.judgement = {
            'vocabulary_breakdown': [
                ['gato', 'cat', 'noun', False]
            ]
        }
        history.update_words(round)
        self.assertIn('gato', history.missed_words)
        self.assertEqual(history.missed_words['gato']['english'], 'cat')
        self.assertEqual(history.missed_words['gato']['count'], 1)

    def test_update_words_missed_word_increment(self):
        history = History()
        history.missed_words = {'gato': {'english': 'cat', 'count': 2}}
        round = TongueRound("Test", 1, 100)
        round.judgement = {
            'vocabulary_breakdown': [
                ['gato', 'cat', 'noun', False]
            ]
        }
        history.update_words(round)
        self.assertEqual(history.missed_words['gato']['count'], 3)

    def test_update_words_correct_removes_from_missed(self):
        history = History()
        history.missed_words = {'gato': {'english': 'cat', 'count': 2}}
        round = TongueRound("Test", 1, 100)
        round.judgement = {
            'vocabulary_breakdown': [
                ['gato', 'cat', 'noun', True]
            ]
        }
        history.update_words(round)
        self.assertNotIn('gato', history.missed_words)
        self.assertIn('gato', history.correct_words)

    def test_from_dict_deduplicates_correct_words(self):
        data = {
            'correct_words': ['casa', 'gato', 'casa', 'perro', 'gato'],
            'difficulty': 3
        }
        history = History.from_dict(data)
        self.assertEqual(history.correct_words, ['casa', 'gato', 'perro'])

    def test_to_dict_and_from_dict_roundtrip(self):
        history = History()
        history.difficulty = 5
        history.correct_words = ['casa', 'gato']
        history.missed_words = {'perro': {'english': 'dog', 'count': 1}}
        history.total_completed = 10
        history.current_story = "Test story."
        history.story_sentences = ["Sentence one.", "Sentence two."]
        history.level_scores = [80, 90]

        data = history.to_dict()
        restored = History.from_dict(data)

        self.assertEqual(restored.difficulty, 5)
        self.assertEqual(restored.correct_words, ['casa', 'gato'])
        self.assertEqual(restored.missed_words, {'perro': {'english': 'dog', 'count': 1}})
        self.assertEqual(restored.total_completed, 10)
        self.assertEqual(restored.current_story, "Test story.")


class TestHistoryProcessEvaluation(unittest.TestCase):
    """Tests for History.process_evaluation method."""

    def test_process_evaluation_records_score(self):
        history = History()
        history.difficulty = 3
        round = TongueRound("Hola mundo.", 3, 100)
        round.translation = "Hello world"

        judgement = {
            'score': 85,
            'correct_translation': 'Hello world',
            'evaluation': 'Good',
            'vocabulary_breakdown': []
        }

        result = history.process_evaluation(judgement, 50, round)

        self.assertTrue(round.evaluated)
        self.assertEqual(round.judge_ms, 50)
        self.assertIn(85, history.level_scores)
        self.assertEqual(history.total_completed, 1)
        self.assertFalse(result['level_changed'])

    def test_process_evaluation_triggers_advancement(self):
        history = History()
        history.difficulty = 3
        history.level_scores = [90, 85, 90, 95]  # 4 good, need 1 more

        round = TongueRound("Test", 3, 100)
        round.translation = "Test"

        judgement = {
            'score': 90,
            'correct_translation': 'Test',
            'evaluation': 'Good',
            'vocabulary_breakdown': []
        }

        result = history.process_evaluation(judgement, 50, round)

        self.assertEqual(history.difficulty, 4)  # Advanced
        self.assertTrue(result['level_changed'])
        self.assertEqual(result['change_type'], 'advanced')

    def test_process_evaluation_triggers_demotion(self):
        history = History()
        history.difficulty = 5
        history.level_scores = [40, 30, 45, 35]  # 4 poor, need 1 more

        round = TongueRound("Test", 5, 100)
        round.translation = "Wrong"

        judgement = {
            'score': 30,
            'correct_translation': 'Test',
            'evaluation': 'Poor',
            'vocabulary_breakdown': []
        }

        result = history.process_evaluation(judgement, 50, round)

        self.assertEqual(history.difficulty, 4)  # Demoted
        self.assertTrue(result['level_changed'])
        self.assertEqual(result['change_type'], 'demoted')


class TestMockAIProvider(unittest.TestCase):
    """Tests for MockAIProvider to ensure it works correctly."""

    def test_generate_story_records_calls(self):
        provider = MockAIProvider()
        provider.set_story_response("Test story.", 100)

        result = provider.generate_story(['casa'], 3)

        self.assertEqual(result, ("Test story.", 100))
        self.assertEqual(len(provider.generate_story_calls), 1)
        self.assertEqual(provider.generate_story_calls[0], (['casa'], 3))

    def test_validate_translation_records_calls(self):
        provider = MockAIProvider()
        provider.set_validation_response({'score': 90}, 50)

        result = provider.validate_translation("Hola", "Hello")

        self.assertEqual(result, ({'score': 90}, 50))
        self.assertEqual(len(provider.validate_translation_calls), 1)
        self.assertEqual(provider.validate_translation_calls[0], ("Hola", "Hello"))

    def test_default_responses(self):
        provider = MockAIProvider()

        story, ms = provider.generate_story([], 1)
        self.assertIn("Default story", story)

        judgement, ms = provider.validate_translation("test", "test")
        self.assertEqual(judgement['score'], 80)

        hint = provider.get_hint("test", [])
        self.assertIsNotNone(hint)


class TestGeminiProviderSanitize(unittest.TestCase):
    """Tests for GeminiProvider response sanitization and error handling."""

    def test_sanitize_judgement_valid_response(self):
        from server.gemini_provider import GeminiProvider
        # Create provider with mock - won't actually connect
        provider = GeminiProvider.__new__(GeminiProvider)

        raw = """Here is the result:
        {'score': 85, 'correct_translation': 'Hello', 'evaluation': 'Good', 'vocabulary_breakdown': []}
        """
        result = provider._sanitize_judgement(raw)
        self.assertIn("'score': 85", result)

    def test_sanitize_judgement_with_booleans(self):
        from server.gemini_provider import GeminiProvider
        provider = GeminiProvider.__new__(GeminiProvider)

        raw = "{'correct': true, 'wrong': false}"
        result = provider._sanitize_judgement(raw)
        self.assertIn("True", result)
        self.assertIn("False", result)

    def test_validate_translation_malformed_response(self):
        """Test that malformed responses return fallback judgement."""
        from server.gemini_provider import GeminiProvider
        from unittest.mock import patch

        provider = GeminiProvider.__new__(GeminiProvider)

        # Mock _execute_chat to return malformed response
        with patch.object(provider, '_execute_chat', return_value=("This is not valid JSON or dict", 100)):
            with patch.object(provider, '_sanitize_judgement', return_value="invalid python"):
                judgement, ms = provider.validate_translation("Hola", "Hello")

        # Should return fallback response
        self.assertEqual(judgement['score'], 50)
        self.assertIn('Error', judgement['evaluation'])
        self.assertEqual(judgement['vocabulary_breakdown'], [])

    def test_validate_translation_valid_response(self):
        """Test that valid responses are parsed correctly."""
        from server.gemini_provider import GeminiProvider
        from unittest.mock import patch

        provider = GeminiProvider.__new__(GeminiProvider)

        valid_dict = "{'score': 95, 'correct_translation': 'Hello world', 'evaluation': 'Excellent', 'vocabulary_breakdown': []}"

        with patch.object(provider, '_execute_chat', return_value=(valid_dict, 100)):
            with patch.object(provider, '_sanitize_judgement', return_value=valid_dict):
                judgement, ms = provider.validate_translation("Hola mundo", "Hello world")

        self.assertEqual(judgement['score'], 95)
        self.assertEqual(judgement['correct_translation'], 'Hello world')


class TestMockStorage(unittest.TestCase):
    """Tests for MockStorage to ensure it works correctly."""

    def test_load_config(self):
        storage = MockStorage()
        storage.set_config({'gemini_api_key': 'my-key'})

        config = storage.load_config()

        self.assertEqual(config['gemini_api_key'], 'my-key')

    def test_load_state_none(self):
        storage = MockStorage()
        state = storage.load_state()
        self.assertIsNone(state)

    def test_load_state_existing(self):
        storage = MockStorage()
        storage.set_state({'difficulty': 5})

        state = storage.load_state()

        self.assertEqual(state['difficulty'], 5)

    def test_save_state_records_calls(self):
        storage = MockStorage()

        storage.save_state({'difficulty': 3})
        storage.save_state({'difficulty': 4})

        self.assertEqual(len(storage.save_calls), 2)
        self.assertEqual(storage.save_calls[0]['difficulty'], 3)
        self.assertEqual(storage.save_calls[1]['difficulty'], 4)


if __name__ == '__main__':
    unittest.main()

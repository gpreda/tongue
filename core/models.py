"""Domain models for tongue application."""

import random

from .config import (
    MIN_DIFFICULTY, MAX_DIFFICULTY,
    ADVANCE_WINDOW_SIZE, ADVANCE_SCORE_THRESHOLD, ADVANCE_REQUIRED_GOOD,
    DEMOTE_SCORE_THRESHOLD, DEMOTE_REQUIRED_POOR,
    CHALLENGE_PROBABILITY
)
from .utils import split_into_sentences


class TongueRound:
    """Represents a single translation round."""

    def __init__(self, sentence: str, difficulty: int, generate_ms: int):
        self.sentence = sentence
        self.difficulty = difficulty
        self.generate_ms = generate_ms
        self.judge_ms = None
        self.translation = None
        self.judgement = None
        self.evaluated = False
        self.raw_ai_response = None

    def to_dict(self) -> dict:
        return {
            'sentence': self.sentence,
            'difficulty': self.difficulty,
            'generate_ms': self.generate_ms,
            'judge_ms': self.judge_ms,
            'translation': self.translation,
            'judgement': self.judgement,
            'evaluated': self.evaluated,
            'raw_ai_response': self.raw_ai_response
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TongueRound':
        round = cls(data['sentence'], data['difficulty'], data['generate_ms'])
        round.judge_ms = data.get('judge_ms')
        round.translation = data.get('translation')
        round.judgement = data.get('judgement')
        round.evaluated = data.get('evaluated', False)
        round.raw_ai_response = data.get('raw_ai_response')
        return round

    def get_score(self) -> int:
        """Get the score from judgement."""
        if self.judgement:
            return int(self.judgement['score'])
        return 0


class History:
    """Tracks user progress and game state."""

    def __init__(self):
        self.rounds = []
        self.correct_words = []  # Legacy, kept for migration
        self.missed_words = {}   # Legacy, kept for migration
        self.words = {}  # New unified word tracking: {word: {type, translation, correct_count, incorrect_count}}
        self.difficulty = MIN_DIFFICULTY
        self.level_scores = []
        self.total_completed = 0
        self.current_story = None
        self.story_sentences = []
        self.story_difficulty = None
        self.story_generate_ms = 0
        self.last_evaluated_round = None
        self.last_level_changed = False
        self.review_queue = []  # Sentences to review: [{sentence, difficulty, due_at_round}]
        # Vocabulary challenge progress: {category: {english_key: {correct: int, incorrect: int}}}
        self.vocab_progress = {}
        self.vocab_progress_version = 2  # Version 2 = english keys
        # Challenge stats (separate from level progress)
        self.challenge_stats = {
            'word': {'correct': 0, 'incorrect': 0},
            'vocab': {'correct': 0, 'incorrect': 0},
            'verb': {'correct': 0, 'incorrect': 0},
            'synonym': {'correct': 0, 'incorrect': 0},
            'weakwords': {'correct': 0, 'incorrect': 0}
        }
        # Story bank: saves unfinished stories per difficulty level for resuming later
        self._story_bank = {}  # difficulty (str) -> {current_story, story_sentences, story_difficulty, story_generate_ms}
        # Practice time tracking per language+direction key, e.g. {"es:normal": 120.5}
        self.practice_times = {}
        # Direction mode: 'normal' (ES→EN) or 'reverse' (EN→ES)
        self.direction = 'normal'
        self._reverse_state = None
        # Language support
        self.language = 'es'
        self._language_states = {}

    @property
    def practice_time_seconds(self) -> float:
        """Total practice time across all languages and directions (backward compat)."""
        return sum(self.practice_times.values())

    @property
    def current_practice_time_seconds(self) -> float:
        """Practice time for the current language:direction."""
        key = f"{self.language}:{self.direction}"
        return self.practice_times.get(key, 0)

    # Fields that are swapped per-direction (everything except practice_times)
    _DIRECTION_FIELDS = [
        'rounds', 'correct_words', 'missed_words', 'words', 'difficulty',
        'level_scores', 'total_completed', 'current_story', 'story_sentences',
        'story_difficulty', 'story_generate_ms', 'last_evaluated_round',
        'last_level_changed', 'review_queue', 'vocab_progress',
        'vocab_progress_version', 'challenge_stats', '_story_bank'
    ]

    # Fields captured per-language (direction + reverse state + all progress)
    _LANGUAGE_FIELDS = _DIRECTION_FIELDS + ['direction', '_reverse_state']

    def switch_direction(self) -> None:
        """Switch between normal (ES→EN) and reverse (EN→ES) modes.
        Captures current direction's state and loads the other direction's state."""
        # Capture current state
        current_state = {}
        for field in self._DIRECTION_FIELDS:
            value = getattr(self, field)
            if field == 'rounds':
                value = [r.to_dict() for r in value]
            elif field == 'last_evaluated_round':
                value = value.to_dict() if value else None
            current_state[field] = value

        # Load other direction's state (or fresh defaults)
        other_state = self._reverse_state
        if other_state:
            for field in self._DIRECTION_FIELDS:
                if field in other_state:
                    value = other_state[field]
                    if field == 'rounds':
                        value = [TongueRound.from_dict(r) for r in value]
                    elif field == 'last_evaluated_round':
                        value = TongueRound.from_dict(value) if value else None
                    setattr(self, field, value)
                else:
                    # Use fresh default for missing fields
                    fresh = History()
                    setattr(self, field, getattr(fresh, field))
        else:
            # First switch — initialize with fresh defaults
            fresh = History()
            for field in self._DIRECTION_FIELDS:
                setattr(self, field, getattr(fresh, field))

        # Save captured state as reverse, flip direction
        self._reverse_state = current_state
        self.direction = 'reverse' if self.direction == 'normal' else 'normal'

    def switch_language(self, new_lang: str) -> None:
        """Switch to a different language, preserving per-language progress.
        Captures current language state, loads target language state or fresh defaults."""
        if new_lang == self.language:
            return

        # Capture current language state
        current_state = {}
        for field in self._LANGUAGE_FIELDS:
            value = getattr(self, field)
            if field == 'rounds':
                value = [r.to_dict() for r in value]
            elif field == 'last_evaluated_round':
                value = value.to_dict() if value else None
            elif field == '_reverse_state':
                value = self._serialize_reverse_state()
            current_state[field] = value
        self._language_states[self.language] = current_state

        # Load target language state or fresh defaults
        target_state = self._language_states.get(new_lang)
        if target_state:
            for field in self._LANGUAGE_FIELDS:
                if field in target_state:
                    value = target_state[field]
                    if field == 'rounds':
                        value = [TongueRound.from_dict(r) for r in value]
                    elif field == 'last_evaluated_round':
                        value = TongueRound.from_dict(value) if value else None
                    setattr(self, field, value)
                else:
                    fresh = History()
                    setattr(self, field, getattr(fresh, field))
            # Remove loaded state from dict (it's now live)
            del self._language_states[new_lang]
        else:
            # First time switching to this language - fresh defaults
            fresh = History()
            for field in self._LANGUAGE_FIELDS:
                setattr(self, field, getattr(fresh, field))

        self.language = new_lang

    def to_dict(self) -> dict:
        return {
            'rounds': [r.to_dict() for r in self.rounds],
            'correct_words': self.correct_words,
            'missed_words': self.missed_words,
            'words': self.words,
            'difficulty': self.difficulty,
            'level_scores': self.level_scores,
            'total_completed': self.total_completed,
            'current_story': self.current_story,
            'story_sentences': self.story_sentences,
            'story_difficulty': self.story_difficulty,
            'story_generate_ms': self.story_generate_ms,
            'review_queue': self.review_queue,
            '_story_bank': self._story_bank,
            'vocab_progress': self.vocab_progress,
            'vocab_progress_version': self.vocab_progress_version,
            'challenge_stats': self.challenge_stats,
            'last_evaluated_round': self.last_evaluated_round.to_dict() if self.last_evaluated_round else None,
            'last_level_changed': self.last_level_changed,
            'practice_times': self.practice_times,
            'direction': self.direction,
            '_reverse_state': self._serialize_reverse_state(),
            'language': self.language,
            '_language_states': self._language_states
        }

    def _serialize_reverse_state(self) -> dict | None:
        """Serialize _reverse_state for persistence."""
        if self._reverse_state is None:
            return None
        state = dict(self._reverse_state)
        # rounds and last_evaluated_round are already dicts when captured by switch_direction
        return state

    @classmethod
    def from_dict(cls, data: dict) -> 'History':
        history = cls()
        history.rounds = [TongueRound.from_dict(r) for r in data.get('rounds', [])]
        # Deduplicate correct_words (legacy)
        seen = set()
        correct_words = []
        for word in data.get('correct_words', []):
            if word not in seen:
                seen.add(word)
                correct_words.append(word)
        history.correct_words = correct_words
        history.missed_words = data.get('missed_words', {})
        history.words = data.get('words', {})

        # Migrate legacy data to new words structure if needed
        if not history.words and (history.correct_words or history.missed_words):
            history._migrate_legacy_words()

        history.difficulty = data.get('difficulty', MIN_DIFFICULTY)
        history.level_scores = data.get('level_scores', [])
        history.total_completed = data.get('total_completed', 0)
        history.current_story = data.get('current_story')
        history.story_sentences = data.get('story_sentences', [])
        history.story_difficulty = data.get('story_difficulty')
        history.story_generate_ms = data.get('story_generate_ms', 0)
        history.review_queue = data.get('review_queue', [])
        history._story_bank = data.get('_story_bank', {})
        history.vocab_progress = data.get('vocab_progress', {})
        history.vocab_progress_version = data.get('vocab_progress_version', 1)
        # Migrate vocab_progress keys from Spanish to English if needed
        if history.vocab_progress_version < 2 and history.vocab_progress:
            history._migrate_vocab_progress_keys()
            history.vocab_progress_version = 2
        history.challenge_stats = data.get('challenge_stats', {
            'word': {'correct': 0, 'incorrect': 0},
            'vocab': {'correct': 0, 'incorrect': 0},
            'verb': {'correct': 0, 'incorrect': 0},
            'synonym': {'correct': 0, 'incorrect': 0},
            'weakwords': {'correct': 0, 'incorrect': 0}
        })
        last_eval = data.get('last_evaluated_round')
        history.last_evaluated_round = TongueRound.from_dict(last_eval) if last_eval else None
        history.last_level_changed = data.get('last_level_changed', False)
        # Migrate from old scalar practice_time_seconds to per-key practice_times dict
        if 'practice_times' in data and isinstance(data['practice_times'], dict):
            history.practice_times = data['practice_times']
        elif 'practice_time_seconds' in data and isinstance(data['practice_time_seconds'], (int, float)):
            old_val = data['practice_time_seconds']
            if old_val > 0:
                # Migrate old scalar into the default language+direction key
                lang = data.get('language', 'es')
                direction = data.get('direction', 'normal')
                history.practice_times = {f"{lang}:{direction}": old_val}
            else:
                history.practice_times = {}
        else:
            history.practice_times = {}
        history.direction = data.get('direction', 'normal')
        history._reverse_state = data.get('_reverse_state', None)
        history.language = data.get('language', 'es')
        history._language_states = data.get('_language_states', {})
        # One-time cleanup: fix words stored with English keys due to safety-net bug
        if history.words:
            history._cleanup_swapped_words()
        return history

    def _migrate_legacy_words(self) -> None:
        """Migrate legacy correct_words and missed_words to unified words structure."""
        # Migrate correct_words (we don't have type/translation info, so use defaults)
        for word in self.correct_words:
            if word not in self.words:
                self.words[word] = {
                    'type': 'unknown',
                    'translation': '',
                    'correct_count': 1,
                    'incorrect_count': 0
                }

        # Migrate missed_words
        for word, info in self.missed_words.items():
            if word in self.words:
                self.words[word]['incorrect_count'] += info.get('count', 1)
                if info.get('english'):
                    self.words[word]['translation'] = info['english']
            else:
                self.words[word] = {
                    'type': 'unknown',
                    'translation': info.get('english', ''),
                    'correct_count': 0,
                    'incorrect_count': info.get('count', 1)
                }

    def _cleanup_swapped_words(self) -> None:
        """Fix words stored with English keys due to inverted safety-net bug.

        A bug in the vocabulary_breakdown safety net caused English translations
        to be stored as word keys (with the target-language word as the translation).
        This detects and fixes those entries by checking if a word's translation
        already exists as another key in the words dict.
        """
        to_delete = []
        to_merge = {}  # target_word -> list of english entries to merge from

        for word, info in list(self.words.items()):
            trans = info.get('translation', '')
            if isinstance(trans, list):
                # Can't reliably detect swap for list translations
                continue
            if not trans or trans.lower().strip() == word.lower().strip():
                continue
            # Check if this word's translation exists as another key
            # AND that other key's translation matches this word
            # This is a strong signal that word/translation are swapped
            trans_stripped = trans.strip()
            if trans_stripped in self.words:
                other = self.words[trans_stripped]
                other_trans = other.get('translation', '')
                if isinstance(other_trans, list):
                    other_trans_str = ', '.join(str(t) for t in other_trans)
                else:
                    other_trans_str = other_trans or ''
                # If the other entry's translation contains this word, they're mirrors
                if word.lower() in other_trans_str.lower():
                    # The trans_stripped key is the correct target-language entry
                    # Merge counts from this (corrupted) entry into it
                    to_merge.setdefault(trans_stripped, []).append(word)
                    to_delete.append(word)

        for target_word, english_keys in to_merge.items():
            for eng_key in english_keys:
                eng_info = self.words[eng_key]
                self.words[target_word]['incorrect_count'] += eng_info.get('incorrect_count', 0)
                self.words[target_word]['correct_count'] += eng_info.get('correct_count', 0)
                if eng_info.get('challenge_passed') is False:
                    self.words[target_word]['challenge_passed'] = False

        for word in to_delete:
            del self.words[word]
            # Also clean up legacy structures
            if word in self.missed_words:
                del self.missed_words[word]
            if word in self.correct_words:
                self.correct_words.remove(word)

    def _migrate_vocab_progress_keys(self) -> None:
        """Migrate vocab_progress from Spanish word keys to English keys.

        Old format: {category: {spanish_word: {correct, incorrect}}}
        New format: {category: {english_key: {correct, incorrect}}}
        """
        from core.vocabulary import VOCABULARY_CHALLENGES

        new_progress = {}
        for category, words in self.vocab_progress.items():
            cat_data = VOCABULARY_CHALLENGES.get(category, {}).get('items', {})
            new_progress[category] = {}
            for old_key, stats in words.items():
                # old_key is a Spanish word — look up its English translation
                alternatives = cat_data.get(old_key)
                if alternatives:
                    english_key = alternatives.split(',')[0].strip()
                    new_progress[category][english_key] = stats
                else:
                    # If we can't find it, keep the old key (it might already be English)
                    new_progress[category][old_key] = stats
        self.vocab_progress = new_progress

    def record_score(self, difficulty: int, score: int, credit: float = 1.0) -> None:
        """Record a score for advancement tracking.

        Args:
            difficulty: The difficulty level
            score: The translation score (0-100)
            credit: Credit multiplier (1.0 for no hint, 0.5 for hint used)
        """
        if difficulty == self.difficulty:
            # Store as [score, credit] for tracking
            self.level_scores.append([score, credit])
            if len(self.level_scores) > ADVANCE_WINDOW_SIZE:
                self.level_scores = self.level_scores[-ADVANCE_WINDOW_SIZE:]

    def get_good_score_count(self) -> float:
        """Get total credits from good scores (>=80). Returns float for partial credits."""
        total = 0.0
        for entry in self.level_scores:
            # Handle both old format (int) and new format ([score, credit])
            if isinstance(entry, list):
                score, credit = entry
            else:
                score, credit = entry, 1.0
            if score >= ADVANCE_SCORE_THRESHOLD:
                total += credit
        return total

    def get_poor_score_count(self) -> int:
        """Get count of poor scores (<50)."""
        count = 0
        for entry in self.level_scores:
            # Handle both old format (int) and new format ([score, credit])
            score = entry[0] if isinstance(entry, list) else entry
            if score < DEMOTE_SCORE_THRESHOLD:
                count += 1
        return count

    def check_advancement(self) -> bool:
        if self.difficulty >= MAX_DIFFICULTY:
            return False
        if len(self.level_scores) >= ADVANCE_WINDOW_SIZE:
            if self.get_good_score_count() >= ADVANCE_REQUIRED_GOOD:
                return True
        return False

    def check_demotion(self) -> bool:
        if self.difficulty <= MIN_DIFFICULTY:
            return False
        if len(self.level_scores) >= ADVANCE_WINDOW_SIZE:
            if self.get_poor_score_count() >= DEMOTE_REQUIRED_POOR:
                return True
        return False

    def save_story_to_bank(self) -> None:
        """Save current story to bank before level change, if there are remaining sentences."""
        if self.story_sentences and self.story_difficulty is not None:
            key = str(self.story_difficulty)
            self._story_bank[key] = {
                'current_story': self.current_story,
                'story_sentences': list(self.story_sentences),
                'story_difficulty': self.story_difficulty,
                'story_generate_ms': self.story_generate_ms
            }

    def load_story_from_bank(self, difficulty: int) -> bool:
        """Restore story for a given difficulty from bank. Returns True if loaded."""
        key = str(difficulty)
        if key in self._story_bank:
            entry = self._story_bank.pop(key)
            self.current_story = entry['current_story']
            self.story_sentences = entry['story_sentences']
            self.story_difficulty = entry['story_difficulty']
            self.story_generate_ms = entry['story_generate_ms']
            return True
        return False

    def advance_level(self) -> bool:
        if self.difficulty < MAX_DIFFICULTY:
            self.save_story_to_bank()
            self.difficulty += 1
            self.level_scores = []
            self.story_sentences = []
            self.current_story = None
            self.last_evaluated_round = None
            # Remove any unevaluated round from previous level
            if self.rounds and hasattr(self.rounds[-1], 'evaluated') and not self.rounds[-1].evaluated:
                self.rounds.pop()
            return True
        return False

    def demote_level(self) -> bool:
        if self.difficulty > MIN_DIFFICULTY:
            self.save_story_to_bank()
            self.difficulty -= 1
            self.level_scores = []
            self.story_sentences = []
            self.current_story = None
            self.last_evaluated_round = None
            # Remove any unevaluated round from previous level
            if self.rounds and hasattr(self.rounds[-1], 'evaluated') and not self.rounds[-1].evaluated:
                self.rounds.pop()
            return True
        return False

    def get_progress_display(self) -> str:
        good_count = self.get_good_score_count()
        attempts = len(self.level_scores)
        if self.difficulty >= MAX_DIFFICULTY:
            return f"Level {self.difficulty}/{MAX_DIFFICULTY} (MAX)"
        return f"Level {self.difficulty}/{MAX_DIFFICULTY} | Progress: {good_count}/{ADVANCE_REQUIRED_GOOD} good scores (last {attempts}/{ADVANCE_WINDOW_SIZE})"

    def needs_new_story(self) -> bool:
        if not self.story_sentences:
            return True
        if self.story_difficulty != self.difficulty:
            return True
        return False

    def reset_story(self) -> None:
        """Clear the current story so a new one will be generated."""
        self.story_sentences = []
        self.current_story = None
        self.story_difficulty = None
        self.story_generate_ms = 0
        # Remove any unevaluated round from the current story
        if self.rounds and hasattr(self.rounds[-1], 'evaluated') and not self.rounds[-1].evaluated:
            self.rounds.pop()

    def set_story(self, story: str, difficulty: int, generate_ms: int) -> None:
        """Set a new story."""
        self.current_story = story
        self.story_sentences = split_into_sentences(story)
        self.story_difficulty = difficulty
        self.story_generate_ms = generate_ms

    def get_next_sentence(self) -> tuple[TongueRound | None, bool]:
        """Get next sentence from story or review queue.
        Returns (round, is_review) tuple. Round is None if no sentences available."""
        # Check for due review sentences first
        for i, review in enumerate(self.review_queue):
            if review['due_at_round'] <= self.total_completed:
                sentence = review['sentence']
                # Skip challenge items that shouldn't be in review queue
                if (sentence.startswith('WORD:') or sentence.startswith('VOCAB:') or
                        sentence.startswith('VOCAB4:') or sentence.startswith('VOCAB4R:') or
                        sentence.startswith('VERB:') or sentence.startswith('SYN:') or
                        sentence.startswith('ANT:') or sentence.startswith('WEAK6:')):
                    self.review_queue.pop(i)
                    continue
                # Remove from queue and return this sentence
                self.review_queue.pop(i)
                round = TongueRound(sentence, review['difficulty'], 0)
                self.rounds.append(round)
                return (round, True)

        # Otherwise get next sentence from story
        if not self.story_sentences:
            return (None, False)
        sentence = self.story_sentences.pop(0)
        round = TongueRound(sentence, self.story_difficulty, self.story_generate_ms)
        self.rounds.append(round)
        return (round, False)

    def update_words(self, round: TongueRound, hint_words: list[str] = None) -> None:
        """Update word tracking based on round judgement."""
        if not round.judgement or 'vocabulary_breakdown' not in round.judgement:
            return

        hint_words = hint_words or []

        for v_breakdown in round.judgement['vocabulary_breakdown']:
            if not isinstance(v_breakdown, (list, tuple)) or len(v_breakdown) < 4:
                continue
            # In normal mode, breakdown is [target_word, english, pos, correct]
            # In reverse mode, breakdown is [english, target_word, pos, correct]
            if self.direction == 'reverse':
                word = v_breakdown[1]
                english = v_breakdown[0]
            else:
                word = v_breakdown[0]
                english = v_breakdown[1]
            # AI may return lists instead of strings; coerce to str
            if isinstance(word, list):
                word = ' '.join(str(w) for w in word)
            if isinstance(english, list):
                english = ', '.join(str(e) for e in english)
            # AI sometimes returns multiple synonyms as a space-separated string
            # (e.g. "rápidos veloz" for "fast"). Split into separate word entries
            # so each synonym is tracked and challenged individually.
            if word and ' ' in word.strip() and ',' not in word:
                synonym_words = word.strip().split()
                if len(synonym_words) <= 3:  # likely synonyms, not a phrase
                    for sw in synonym_words:
                        sw = sw.strip()
                        if not sw:
                            continue
                        if sw not in self.words:
                            self.words[sw] = {
                                'type': (v_breakdown[2] or 'unknown').lower(),
                                'translation': english,
                                'correct_count': 0,
                                'incorrect_count': 0
                            }
                        if v_breakdown[3]:
                            self.words[sw]['correct_count'] += 1
                        else:
                            self.words[sw]['incorrect_count'] += 1
                            self.words[sw]['challenge_passed'] = False
                    continue
            # Safety net: detect when AI returned vocabulary_breakdown in wrong order.
            # The word (target language) SHOULD appear in the original sentence.
            # If english appears in the sentence but word doesn't, they're likely swapped.
            sentence_lower = (round.sentence or '').lower()
            word_lower = (word or '').lower().strip()
            english_lower = (english or '').lower().strip()
            # Strip punctuation from sentence words for robust matching
            sentence_words = {w.strip('.,;:!?¿¡"\'()[]') for w in sentence_lower.split()}
            if (word_lower and english_lower and word_lower != english_lower
                    and word_lower not in sentence_words
                    and english_lower in sentence_words):
                word, english = english, word
            # Skip entries where word == english (AI returned same value for both,
            # so we can't determine the correct translation)
            if word and english and word.lower().strip() == english.lower().strip():
                continue
            part_of_speech = (v_breakdown[2] or 'unknown').lower()
            was_correct = v_breakdown[3]

            # Words given as hints count as incorrect
            if word in hint_words:
                was_correct = False

            # Update unified words structure
            if word not in self.words:
                self.words[word] = {
                    'type': part_of_speech,
                    'translation': english,
                    'correct_count': 0,
                    'incorrect_count': 0
                }
            else:
                # Update type/translation if we have better info
                if self.words[word]['type'] == 'unknown':
                    self.words[word]['type'] = part_of_speech
                if not self.words[word]['translation']:
                    self.words[word]['translation'] = english

            if was_correct:
                self.words[word]['correct_count'] += 1
            else:
                self.words[word]['incorrect_count'] += 1
                # Reset challenge_passed so the word can reappear in challenges
                self.words[word]['challenge_passed'] = False

            # Also update legacy structures for backwards compatibility
            if was_correct and part_of_speech in ['noun', 'verb']:
                if word not in self.correct_words:
                    self.correct_words.append(word)
                if word in self.missed_words:
                    del self.missed_words[word]
            elif not was_correct:
                if word in self.missed_words:
                    self.missed_words[word]['count'] += 1
                else:
                    self.missed_words[word] = {'english': english, 'count': 1}

    def process_evaluation(self, judgement: dict, judge_ms: int, round: TongueRound, hint_words: list[str] = None, hint_used: bool = False) -> dict:
        """Process evaluation results and return status."""
        round.judgement = judgement
        round.judge_ms = judge_ms
        round.evaluated = True

        score = round.get_score()
        # Hint used = 0.5 credit, no hint = 1.0 credit
        credit = 0.5 if hint_used else 1.0
        self.record_score(round.difficulty, score, credit)
        self.update_words(round, hint_words or [])
        self.total_completed += 1

        # Add low-score sentences to review queue (bring back after 5 rounds)
        # Don't add challenges (WORD:, VOCAB:, VERB:) to review queue
        is_challenge = (round.sentence.startswith('WORD:') or
                        round.sentence.startswith('VOCAB:') or
                        round.sentence.startswith('VOCAB4:') or
                        round.sentence.startswith('VOCAB4R:') or
                        round.sentence.startswith('VERB:') or
                        round.sentence.startswith('SYN:') or
                        round.sentence.startswith('ANT:') or
                        round.sentence.startswith('WEAK6:'))
        if score <= 50 and not is_challenge:
            # Check if sentence is not already in review queue
            existing = [r['sentence'] for r in self.review_queue]
            if round.sentence not in existing:
                self.review_queue.append({
                    'sentence': round.sentence,
                    'difficulty': round.difficulty,
                    'due_at_round': self.total_completed + 5
                })

        level_changed = False
        new_level = self.difficulty
        change_type = None

        if self.check_advancement():
            self.advance_level()
            level_changed = True
            new_level = self.difficulty
            change_type = "advanced"
        elif self.check_demotion():
            self.demote_level()
            level_changed = True
            new_level = self.difficulty
            change_type = "demoted"

        self.last_evaluated_round = round
        self.last_level_changed = level_changed

        return {
            'level_changed': level_changed,
            'new_level': new_level,
            'change_type': change_type
        }

    def get_mastered_words(self) -> list[dict]:
        """Get words with success rate >= 80% and at least 2 correct.
        Returns list sorted by total frequency (correct + incorrect) descending."""
        mastered = []
        for word, info in self.words.items():
            total = info['correct_count'] + info['incorrect_count']
            if total > 0 and info['correct_count'] >= 2:
                success_rate = info['correct_count'] / total
                if success_rate >= 0.8:
                    mastered.append({
                        'word': word,
                        'type': info['type'],
                        'translation': info['translation'],
                        'correct_count': info['correct_count'],
                        'incorrect_count': info['incorrect_count'],
                        'total': total,
                        'success_rate': round(success_rate * 100, 1)
                    })
        return sorted(mastered, key=lambda x: x['total'], reverse=True)

    def get_learning_words(self) -> list[dict]:
        """Get words with success rate <= 50%.
        Returns list sorted by total frequency (correct + incorrect) descending."""
        learning = []
        for word, info in self.words.items():
            total = info['correct_count'] + info['incorrect_count']
            if total > 0:
                success_rate = info['correct_count'] / total
                if success_rate <= 0.5:
                    learning.append({
                        'word': word,
                        'type': info['type'],
                        'translation': info['translation'],
                        'correct_count': info['correct_count'],
                        'incorrect_count': info['incorrect_count'],
                        'total': total,
                        'success_rate': round(success_rate * 100, 1)
                    })
        return sorted(learning, key=lambda x: x['total'], reverse=True)

    def is_challenge_turn(self) -> bool:
        """Roll whether this turn should be a challenge (~30% chance)."""
        return random.random() < CHALLENGE_PROBABILITY

    def get_challenge_category_weights(self) -> dict[str, float]:
        """Return {category: weight} for eligible categories based on success rates.
        Weight = 100 - success_rate. Categories with 0 attempts get weight 100."""
        weights = {}
        # word: need at least 3 words
        if len(self.words) >= 3:
            weights['word'] = self._challenge_weight('word')
        # vocab: always eligible
        weights['vocab'] = self._challenge_weight('vocab')
        # verb: need at least 1 verb
        has_verb = any(
            (info.get('type') or '').lower() == 'verb'
            for info in self.words.values()
        )
        if has_verb:
            weights['verb'] = self._challenge_weight('verb')
        # synonym: only in reverse mode (L1→L2), need >= 5 words and at least 1 noun/verb/adjective
        if self.direction == 'reverse' and len(self.words) >= 5:
            has_eligible = any(
                (info.get('type') or '').lower() in ('noun', 'verb', 'adjective')
                for info in self.words.values()
            )
            if has_eligible:
                weights['synonym'] = self._challenge_weight('synonym')
        # weakwords: need >= 6 words with valid translations (non-proper-noun)
        if self.get_weakest_words(6) is not None:
            weights['weakwords'] = self._challenge_weight('weakwords')
        return weights

    def _challenge_weight(self, category: str) -> float:
        """Compute weight for a challenge category: 100 - success_rate.
        No attempts = success_rate 0% = weight 100."""
        stats = self.challenge_stats.get(category, {'correct': 0, 'incorrect': 0})
        total = stats['correct'] + stats['incorrect']
        if total == 0:
            return 100.0
        success_rate = (stats['correct'] / total) * 100
        return max(100.0 - success_rate, 0.0)

    def pick_challenge_type(self) -> str | None:
        """Decide whether this turn is a challenge and which category.
        Returns category name ('word', 'vocab', 'verb', 'synonym', 'weakwords') or None."""
        # Every 5th round, force a weakwords challenge if eligible
        round_num = len(self.rounds) + 1  # +1 for the upcoming round
        if round_num % 5 == 0 and self.get_weakest_words(6) is not None:
            return 'weakwords'
        if not self.is_challenge_turn():
            return None
        weights = self.get_challenge_category_weights()
        if not weights:
            return None
        # If all weights are 0, use uniform distribution
        categories = list(weights.keys())
        w = list(weights.values())
        if all(v == 0 for v in w):
            return random.choice(categories)
        return random.choices(categories, weights=w, k=1)[0]

    @staticmethod
    def _is_proper_noun(word: str) -> bool:
        """Check if a word is a proper noun (personal name, place, etc.)."""
        return word[0].isupper() if word else False

    def _has_valid_translation(self, word: str) -> bool:
        """Check if a word has a valid (non-empty, non-self) translation."""
        info = self.words.get(word, {})
        trans = info.get('translation') or ''
        if isinstance(trans, list):
            trans = ', '.join(str(t) for t in trans)
        return bool(trans) and trans.lower().strip() != word.lower().strip()

    def get_challenge_word(self) -> dict | None:
        """Get a word for word challenge. Picks from low success rate words.
        Excludes proper nouns (personal names).
        Returns dict with word, type, translation or None if no words available."""
        # Get words with low success rate (<=70%), excluding proper nouns
        candidates = []
        for word, info in self.words.items():
            if self._is_proper_noun(word):
                continue
            # Skip words already passed in a challenge (unless missed later)
            if info.get('challenge_passed'):
                continue
            # Skip words with missing or invalid translations (translation equals
            # the word itself, which means the AI stored it incorrectly)
            trans = info.get('translation') or ''
            if isinstance(trans, list):
                trans = ', '.join(str(t) for t in trans)
            if not trans or trans.lower().strip() == word.lower().strip():
                continue
            total = info['correct_count'] + info['incorrect_count']
            if total > 0:
                success_rate = info['correct_count'] / total
                if success_rate <= 0.7:
                    candidates.append({
                        'word': word,
                        'type': info['type'] or 'unknown',
                        'translation': info['translation'] or '',
                        'success_rate': success_rate
                    })

        if not candidates:
            # Fall back to any non-proper-noun word if no low success rate words
            eligible = [w for w in self.words.keys()
                        if not self._is_proper_noun(w) and not self.words[w].get('challenge_passed')
                        and self._has_valid_translation(w)]
            if not eligible:
                return None
            word = random.choice(eligible)
            info = self.words[word]
            return {
                'word': word,
                'type': info['type'] or 'unknown',
                'translation': info['translation'] or ''
            }

        # Weight by inverse success rate (lower success = higher chance)
        weights = [(1 - c['success_rate'] + 0.1) for c in candidates]
        chosen = random.choices(candidates, weights=weights, k=1)[0]
        return {
            'word': chosen['word'],
            'type': chosen['type'],
            'translation': chosen['translation']
        }

    def get_vocab_challenge(self) -> dict | None:
        """Get a vocabulary challenge from a random category.
        Each category has equal probability. Each word within a category has equal probability.
        Translation direction: es->en 80%, en->es 20%.
        For multi-word categories (day, month, season, number), returns a multi-word challenge.
        For other categories, returns single-word challenge.
        """
        from core.vocabulary import (get_all_categories, get_random_challenge,
                                     get_multi_word_challenge, MULTI_WORD_CATEGORIES)

        lang = getattr(self, 'language', 'es')
        category = random.choice(get_all_categories(lang))
        reverse = random.random() < 0.2  # 20% en->target, 80% target->en

        if category in MULTI_WORD_CATEGORIES:
            return get_multi_word_challenge(category, reverse=reverse, language=lang)
        else:
            return get_random_challenge(category, reverse=reverse, language=lang)

    def record_vocab_result(self, category: str, english_key: str, is_correct: bool) -> None:
        """Record result of a vocabulary challenge by english key."""
        if category not in self.vocab_progress:
            self.vocab_progress[category] = {}

        if english_key not in self.vocab_progress[category]:
            self.vocab_progress[category][english_key] = {'correct': 0, 'incorrect': 0}

        if is_correct:
            self.vocab_progress[category][english_key]['correct'] += 1
        else:
            self.vocab_progress[category][english_key]['incorrect'] += 1

    def get_word_for_synonym_challenge(self) -> dict | None:
        """Get a word from user's learned words for synonym/antonym challenge.
        Only picks nouns, verbs, adjectives (skips proper nouns).
        Returns dict with word, type, translation or None."""
        candidates = []
        for word, info in self.words.items():
            if self._is_proper_noun(word):
                continue
            word_type = (info.get('type') or '').lower()
            if word_type not in ('noun', 'verb', 'adjective'):
                continue
            candidates.append({
                'word': word,
                'type': word_type,
                'translation': info.get('translation', '')
            })

        if not candidates:
            return None
        return random.choice(candidates)

    def get_weakest_words(self, count: int = 6) -> list[dict] | None:
        """Get words with the lowest success rate from user's word history.
        Excludes proper nouns and words without valid translations.
        Returns list of {word, type, translation} or None if fewer than count eligible."""
        candidates = []
        for word, info in self.words.items():
            if self._is_proper_noun(word):
                continue
            if not self._has_valid_translation(word):
                continue
            total = info['correct_count'] + info['incorrect_count']
            if total == 0:
                success_rate = 0.0
            else:
                success_rate = info['correct_count'] / total
            trans = info.get('translation', '')
            if isinstance(trans, list):
                trans = ', '.join(str(t) for t in trans)
            candidates.append({
                'word': word,
                'type': info.get('type') or 'unknown',
                'translation': trans,
                'success_rate': success_rate,
                'total': total
            })

        if len(candidates) < count:
            return None

        # Sort by success rate ascending (weakest first), then by total descending (more practiced = more confident about weakness)
        candidates.sort(key=lambda c: (c['success_rate'], -c['total']))
        result = candidates[:count]
        # Remove internal keys before returning
        for item in result:
            del item['success_rate']
            del item['total']
        return result

    def get_verb_for_challenge(self) -> str | None:
        """Get a verb from user's word history for verb challenge.
        Returns the target-language verb or None if no verbs available."""
        verbs = []
        for word, info in self.words.items():
            word_type = (info.get('type') or '').lower()
            if word_type == 'verb':
                verbs.append(word)

        if not verbs:
            return None

        return random.choice(verbs)

    def record_challenge_result(self, challenge_type: str, is_correct: bool) -> None:
        """Record result of a challenge (word, vocab, or verb).
        This is separate from level progress."""
        if challenge_type not in self.challenge_stats:
            self.challenge_stats[challenge_type] = {'correct': 0, 'incorrect': 0}

        if is_correct:
            self.challenge_stats[challenge_type]['correct'] += 1
        else:
            self.challenge_stats[challenge_type]['incorrect'] += 1

    def get_challenge_stats_display(self) -> str:
        """Get a display string for challenge stats as a success percentage."""
        total_correct = sum(s['correct'] for s in self.challenge_stats.values())
        total_attempts = total_correct + sum(s['incorrect'] for s in self.challenge_stats.values())
        if total_attempts == 0:
            return "0%"
        pct = round(total_correct / total_attempts * 100)
        return f"{pct}%"

    def record_practice_time(self, delta_seconds: float) -> bool:
        """Record practice time if within threshold. Returns True if recorded."""
        from core.config import PRACTICE_TIME_INACTIVITY_THRESHOLD
        if 0 < delta_seconds <= PRACTICE_TIME_INACTIVITY_THRESHOLD:
            key = f"{self.language}:{self.direction}"
            self.practice_times[key] = self.practice_times.get(key, 0) + delta_seconds
            return True
        return False

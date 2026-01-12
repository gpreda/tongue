"""Domain models for tongue application."""

import random

from .config import (
    MIN_DIFFICULTY, MAX_DIFFICULTY,
    ADVANCE_WINDOW_SIZE, ADVANCE_SCORE_THRESHOLD, ADVANCE_REQUIRED_GOOD,
    DEMOTE_SCORE_THRESHOLD, DEMOTE_REQUIRED_POOR
)
from .utils import split_into_sentences

WORD_CHALLENGE_INTERVAL = 3  # Every 3rd turn


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

    def to_dict(self) -> dict:
        return {
            'sentence': self.sentence,
            'difficulty': self.difficulty,
            'generate_ms': self.generate_ms,
            'judge_ms': self.judge_ms,
            'translation': self.translation,
            'judgement': self.judgement,
            'evaluated': self.evaluated
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TongueRound':
        round = cls(data['sentence'], data['difficulty'], data['generate_ms'])
        round.judge_ms = data.get('judge_ms')
        round.translation = data.get('translation')
        round.judgement = data.get('judgement')
        round.evaluated = data.get('evaluated', False)
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
        # Vocabulary challenge progress: {category: {word: {correct: int, incorrect: int}}}
        self.vocab_progress = {}
        # Challenge stats (separate from level progress)
        self.challenge_stats = {
            'word': {'correct': 0, 'incorrect': 0},
            'vocab': {'correct': 0, 'incorrect': 0},
            'verb': {'correct': 0, 'incorrect': 0}
        }

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
            'vocab_progress': self.vocab_progress,
            'challenge_stats': self.challenge_stats
        }

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
        history.vocab_progress = data.get('vocab_progress', {})
        history.challenge_stats = data.get('challenge_stats', {
            'word': {'correct': 0, 'incorrect': 0},
            'vocab': {'correct': 0, 'incorrect': 0},
            'verb': {'correct': 0, 'incorrect': 0}
        })
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

    def advance_level(self) -> bool:
        if self.difficulty < MAX_DIFFICULTY:
            self.difficulty += 1
            self.level_scores = []
            self.story_sentences = []
            self.current_story = None
            return True
        return False

    def demote_level(self) -> bool:
        if self.difficulty > MIN_DIFFICULTY:
            self.difficulty -= 1
            self.level_scores = []
            self.story_sentences = []
            self.current_story = None
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
                # Remove from queue and return this sentence
                self.review_queue.pop(i)
                round = TongueRound(review['sentence'], review['difficulty'], 0)
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
            word = v_breakdown[0]
            english = v_breakdown[1]
            part_of_speech = (v_breakdown[2] or 'unknown').lower()
            was_correct = v_breakdown[3]

            # Skip words that were given as hints (don't count for stats)
            if word in hint_words:
                continue

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
        if score <= 50:
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

    def is_word_challenge_turn(self) -> bool:
        """Check if this turn should be a word challenge."""
        # Need at least some words to challenge
        if len(self.words) < 3:
            return False
        # Every Nth turn (but not the first few)
        if self.total_completed < WORD_CHALLENGE_INTERVAL:
            return False
        return self.total_completed % WORD_CHALLENGE_INTERVAL == 0

    def get_challenge_word(self) -> dict | None:
        """Get a word for word challenge. Picks from low success rate words.
        Returns dict with word, type, translation or None if no words available."""
        # Get words with low success rate (<=70%)
        candidates = []
        for word, info in self.words.items():
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
            # Fall back to any word if no low success rate words
            if not self.words:
                return None
            word = random.choice(list(self.words.keys()))
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

    def is_vocab_challenge_turn(self) -> bool:
        """Check if this turn should be a vocabulary category challenge."""
        # Every 5th turn (offset from word challenges)
        if self.total_completed < 5:
            return False
        return self.total_completed % 5 == 2  # Offset so it doesn't overlap with word challenges

    def get_vocab_challenge(self) -> dict | None:
        """Get a vocabulary challenge from a random category.
        Returns dict with: word, translation, category, category_name
        Prioritizes words the user hasn't mastered yet."""
        from core.vocabulary import get_all_categories, get_random_challenge

        categories = get_all_categories()
        random.shuffle(categories)

        for category in categories:
            # Get words user has mastered in this category
            mastered = self._get_mastered_vocab_words(category)

            challenge = get_random_challenge(category, exclude_words=mastered)
            if challenge:
                return challenge

        return None

    def _get_mastered_vocab_words(self, category: str) -> list[str]:
        """Get list of words user has mastered in a category (>=2 correct, >=80% rate)."""
        mastered = []
        if category not in self.vocab_progress:
            return mastered

        for word, stats in self.vocab_progress[category].items():
            correct = stats.get('correct', 0)
            incorrect = stats.get('incorrect', 0)
            total = correct + incorrect
            if total >= 2 and correct / total >= 0.8:
                mastered.append(word)

        return mastered

    def record_vocab_result(self, category: str, word: str, is_correct: bool) -> None:
        """Record result of a vocabulary challenge."""
        if category not in self.vocab_progress:
            self.vocab_progress[category] = {}

        if word not in self.vocab_progress[category]:
            self.vocab_progress[category][word] = {'correct': 0, 'incorrect': 0}

        if is_correct:
            self.vocab_progress[category][word]['correct'] += 1
        else:
            self.vocab_progress[category][word]['incorrect'] += 1

    def is_verb_challenge_turn(self) -> bool:
        """Check if this turn should be a verb challenge."""
        # Every 7th turn (offset from other challenges)
        if self.total_completed < 7:
            return False
        return self.total_completed % 7 == 0

    def get_verb_for_challenge(self) -> str | None:
        """Get a verb from user's word history for verb challenge.
        Returns the Spanish verb or None if no verbs available."""
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
        """Get a display string for challenge stats."""
        total_correct = sum(s['correct'] for s in self.challenge_stats.values())
        total_attempts = total_correct + sum(s['incorrect'] for s in self.challenge_stats.values())
        if total_attempts == 0:
            return "0/0"
        return f"{total_correct}/{total_attempts}"

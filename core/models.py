"""Domain models for tongue application."""

from .config import (
    MIN_DIFFICULTY, MAX_DIFFICULTY,
    ADVANCE_WINDOW_SIZE, ADVANCE_SCORE_THRESHOLD, ADVANCE_REQUIRED_GOOD,
    DEMOTE_SCORE_THRESHOLD, DEMOTE_REQUIRED_POOR
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
            'story_generate_ms': self.story_generate_ms
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

    def record_score(self, difficulty: int, score: int) -> None:
        """Record a score for advancement tracking."""
        if difficulty == self.difficulty:
            self.level_scores.append(score)
            if len(self.level_scores) > ADVANCE_WINDOW_SIZE:
                self.level_scores = self.level_scores[-ADVANCE_WINDOW_SIZE:]

    def get_good_score_count(self) -> int:
        return sum(1 for s in self.level_scores if s >= ADVANCE_SCORE_THRESHOLD)

    def get_poor_score_count(self) -> int:
        return sum(1 for s in self.level_scores if s < DEMOTE_SCORE_THRESHOLD)

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

    def get_next_sentence(self) -> TongueRound | None:
        """Get next sentence from story. Returns None if no sentences available."""
        if not self.story_sentences:
            return None
        sentence = self.story_sentences.pop(0)
        round = TongueRound(sentence, self.story_difficulty, self.story_generate_ms)
        self.rounds.append(round)
        return round

    def update_words(self, round: TongueRound, hint_words: list[str] = None) -> None:
        """Update word tracking based on round judgement."""
        if not round.judgement or 'vocabulary_breakdown' not in round.judgement:
            return

        hint_words = hint_words or []

        for v_breakdown in round.judgement['vocabulary_breakdown']:
            word = v_breakdown[0]
            english = v_breakdown[1]
            part_of_speech = v_breakdown[2].lower()
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

    def process_evaluation(self, judgement: dict, judge_ms: int, round: TongueRound, hint_words: list[str] = None) -> dict:
        """Process evaluation results and return status."""
        round.judgement = judgement
        round.judge_ms = judge_ms
        round.evaluated = True

        score = round.get_score()
        self.record_score(round.difficulty, score)
        self.update_words(round, hint_words or [])
        self.total_completed += 1

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

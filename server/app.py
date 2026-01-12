"""FastAPI server for tongue application."""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
from typing import Optional

from core.models import History, TongueRound
from core.config import (
    LANGUAGE, MIN_DIFFICULTY, MAX_DIFFICULTY,
    ADVANCE_SCORE_THRESHOLD, ADVANCE_REQUIRED_GOOD, ADVANCE_WINDOW_SIZE,
    DEMOTE_SCORE_THRESHOLD, DEMOTE_REQUIRED_POOR, STORY_SENTENCE_COUNT
)

from server.gemini_provider import GeminiProvider
from server.file_storage import FileStorage
from server.postgres_storage import PostgresStorage

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
WEB_DIR = PROJECT_ROOT / "web"


# Pydantic models for API
class TranslationRequest(BaseModel):
    sentence: str
    translation: str
    user_id: str = "default"
    hint_used: bool = False
    hint_words: list[str] = []
    selected_tense: Optional[str] = None  # For verb challenges


class HintRequest(BaseModel):
    sentence: str
    user_id: str = "default"


class CreateUserRequest(BaseModel):
    pin: str


class LoginRequest(BaseModel):
    pin: str


class StoryResponse(BaseModel):
    story: str
    sentences: list[str]
    difficulty: int
    generate_ms: int
    current_sentence: Optional[str]
    sentences_remaining: int


class TranslationResponse(BaseModel):
    score: int
    correct_translation: str
    evaluation: str
    vocabulary_breakdown: list
    judge_ms: int
    level_changed: bool
    new_level: int
    change_type: Optional[str]


class HintResponse(BaseModel):
    noun: Optional[list]
    verb: Optional[list]
    adjective: Optional[list]


class StatusResponse(BaseModel):
    language: str
    difficulty: int
    max_difficulty: int
    total_completed: int
    mastered_words_count: int
    missed_words_count: int
    recent_mastered: list[str]
    level_scores: list
    good_score_count: float
    poor_score_count: int
    story_sentences_remaining: int
    progress_display: str
    challenge_stats: dict  # {word: {correct, incorrect}, vocab: {...}, verb: {...}}
    challenge_stats_display: str  # "12/15" format


class NextSentenceResponse(BaseModel):
    sentence: str
    difficulty: int
    story: str
    sentences_remaining: int
    progress_display: str
    is_review: bool
    is_word_challenge: bool
    challenge_word: Optional[dict]  # {word, type, translation} for word challenges
    is_vocab_challenge: bool = False
    vocab_challenge: Optional[dict] = None  # {word, translation, category, category_name}
    is_verb_challenge: bool = False
    verb_challenge: Optional[dict] = None  # {conjugated_form, base_verb, tense, translation, person}
    has_previous_evaluation: bool
    previous_evaluation: Optional[dict]


# Valid tenses for verb challenges
VALID_TENSES = ['present', 'preterite', 'imperfect', 'future', 'conditional', 'subjunctive']


# Global state (in production, use proper DI)
storage: FileStorage = None
ai_provider: GeminiProvider = None
user_histories: dict[str, History] = {}

app = FastAPI(title="Tongue API", description="Spanish translation practice API")

# Mount static files
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/app")
async def serve_app():
    """Serve the web application."""
    return FileResponse(WEB_DIR / "index.html")


def get_history(user_id: str = "default") -> History:
    """Get or create history for a user."""
    if user_id not in user_histories:
        state = storage.load_state(user_id)
        if state:
            user_histories[user_id] = History.from_dict(state)
        else:
            user_histories[user_id] = History()
    return user_histories[user_id]


def save_history(user_id: str = "default") -> None:
    """Save history for a user."""
    if user_id in user_histories:
        storage.save_state(user_histories[user_id].to_dict(), user_id)


@app.on_event("startup")
async def startup():
    """Initialize storage and AI provider on startup."""
    global storage, ai_provider

    # Use PostgreSQL by default, set TONGUE_STORAGE=file to use file storage
    storage_type = os.environ.get('TONGUE_STORAGE', 'postgres')
    if storage_type == 'file':
        storage = FileStorage()
        print("Using file storage")
    else:
        storage = PostgresStorage()
        print("Using PostgreSQL storage")

    # Get API key from environment variable first, then fall back to config file
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        try:
            config = storage.load_config()
            api_key = config.get('gemini_api_key')
        except FileNotFoundError:
            pass

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable not set and config file not found. "
            "Set GEMINI_API_KEY or create ~/.config/tongue/config.json"
        )

    ai_provider = GeminiProvider(api_key)


@app.get("/")
async def root():
    """Redirect root to the app."""
    return RedirectResponse(url="/app")


# User Management Endpoints
@app.get("/api/users")
async def list_users():
    """List all existing users."""
    users = storage.list_users()
    # Filter out 'default' as it's a system user
    users = [u for u in users if u != 'default']
    return {"users": users}


@app.get("/api/users/{user_id}/exists")
async def check_user_exists(user_id: str):
    """Check if a user exists."""
    exists = storage.user_exists(user_id)
    return {"exists": exists}


@app.post("/api/users/{user_id}")
async def create_user(user_id: str, request: CreateUserRequest):
    """Create a new user with a PIN. Returns error if user already exists."""
    if storage.user_exists(user_id):
        return {"success": False, "error": "User already exists"}

    # Validate PIN is 4 digits
    if not request.pin or len(request.pin) != 4 or not request.pin.isdigit():
        return {"success": False, "error": "PIN must be exactly 4 digits"}

    # Create empty history for new user
    history = History()
    user_histories[user_id] = history
    save_history(user_id)

    # Save the PIN
    storage.save_pin(user_id, request.pin)

    return {"success": True, "user_id": user_id}


@app.post("/api/users/{user_id}/login")
async def login_user(user_id: str, request: LoginRequest):
    """Login an existing user with PIN verification."""
    if not storage.user_exists(user_id):
        return {"success": False, "error": "User not found"}

    # Check if user has a PIN set
    existing_pin = storage.get_pin_hash(user_id)

    if not existing_pin:
        # Legacy user without PIN - set their PIN now
        storage.save_pin(user_id, request.pin)
        return {"success": True, "user_id": user_id}

    # Verify PIN
    if not storage.verify_pin(user_id, request.pin):
        return {
            "success": False,
            "error": "User name already exists. Please enter the correct PIN or choose a different name to start a new game."
        }

    return {"success": True, "user_id": user_id}


@app.get("/api/status", response_model=StatusResponse)
async def get_status(user_id: str = "default"):
    """Get user status and progress."""
    history = get_history(user_id)

    return StatusResponse(
        language=LANGUAGE,
        difficulty=history.difficulty,
        max_difficulty=MAX_DIFFICULTY,
        total_completed=history.total_completed,
        mastered_words_count=len(history.correct_words),
        missed_words_count=len(history.missed_words),
        recent_mastered=history.correct_words[-10:] if history.correct_words else [],
        level_scores=history.level_scores,
        good_score_count=history.get_good_score_count(),
        poor_score_count=history.get_poor_score_count(),
        story_sentences_remaining=len(history.story_sentences),
        progress_display=history.get_progress_display(),
        challenge_stats=history.challenge_stats,
        challenge_stats_display=history.get_challenge_stats_display()
    )


@app.get("/api/story", response_model=StoryResponse)
async def get_story(user_id: str = "default", force_new: bool = False):
    """Get current story or generate a new one."""
    history = get_history(user_id)

    if force_new or history.needs_new_story():
        story, ms = ai_provider.generate_story(
            history.correct_words,
            history.difficulty
        )
        history.set_story(story, history.difficulty, ms)
        save_history(user_id)

    current_sentence = None
    if history.story_sentences:
        current_sentence = history.story_sentences[0]

    return StoryResponse(
        story=history.current_story or "",
        sentences=history.story_sentences,
        difficulty=history.story_difficulty or history.difficulty,
        generate_ms=history.story_generate_ms,
        current_sentence=current_sentence,
        sentences_remaining=len(history.story_sentences)
    )


@app.get("/api/next", response_model=NextSentenceResponse)
async def get_next_sentence(user_id: str = "default"):
    """Get the next sentence, word challenge, vocab challenge, or verb challenge."""
    history = get_history(user_id)

    # Check if there's an existing unevaluated round (e.g., after page refresh)
    current_round = None
    is_review = False
    is_word_challenge = False
    is_vocab_challenge = False
    is_verb_challenge = False
    challenge_word = None
    vocab_challenge = None
    verb_challenge = None

    if history.rounds:
        last_round = history.rounds[-1]
        if not last_round.evaluated:
            current_round = last_round
            # Check if this was a review sentence (generate_ms=0 indicates review)
            is_review = last_round.generate_ms == 0
            # Check if this was a word challenge (sentence starts with "WORD:")
            if last_round.sentence.startswith("WORD:"):
                is_word_challenge = True
                word = last_round.sentence[5:]  # Remove "WORD:" prefix
                # Get word info
                if word in history.words:
                    info = history.words[word]
                    challenge_word = {
                        'word': word,
                        'type': info['type'],
                        'translation': info['translation']
                    }
            # Check if this was a vocab challenge (sentence starts with "VOCAB:")
            elif last_round.sentence.startswith("VOCAB:"):
                is_vocab_challenge = True
                # Format: VOCAB:category:word
                parts = last_round.sentence.split(":", 2)
                if len(parts) == 3:
                    from core.vocabulary import get_category_name, get_category_items
                    category = parts[1]
                    word = parts[2]
                    items = get_category_items(category)
                    vocab_challenge = {
                        'word': word,
                        'translation': items.get(word, ''),
                        'category': category,
                        'category_name': get_category_name(category)
                    }
            # Check if this was a verb challenge (sentence starts with "VERB:")
            elif last_round.sentence.startswith("VERB:"):
                is_verb_challenge = True
                conjugated_form = last_round.sentence[5:]  # Remove "VERB:" prefix
                # Get verb info from storage
                stored = storage.get_verb_conjugation(conjugated_form)
                if stored:
                    verb_challenge = {
                        'conjugated_form': conjugated_form,
                        **stored
                    }

    # Only get a new sentence/challenge if there's no current unevaluated round
    if current_round is None:
        # Check if it's verb challenge turn (every 7th turn)
        if history.is_verb_challenge_turn():
            verb_word = history.get_verb_for_challenge()
            if verb_word:
                # Check storage for existing conjugation, or query AI
                stored = storage.get_verb_conjugation(verb_word)
                if stored:
                    verb_challenge = {'conjugated_form': verb_word, **stored}
                else:
                    # Query AI for verb conjugation
                    ai_result = ai_provider.analyze_verb_conjugation(verb_word)
                    if ai_result:
                        storage.save_verb_conjugation(
                            verb_word, ai_result['base_verb'], ai_result['tense'],
                            ai_result['translation'], ai_result['person']
                        )
                        verb_challenge = {'conjugated_form': verb_word, **ai_result}

                if verb_challenge:
                    is_verb_challenge = True
                    from core.models import TongueRound
                    current_round = TongueRound(f"VERB:{verb_word}", history.difficulty, 0)
                    history.rounds.append(current_round)
                    save_history(user_id)

        # Check if it's vocab challenge turn (every 5th turn, offset by 2)
        if current_round is None and history.is_vocab_challenge_turn():
            vocab_challenge = history.get_vocab_challenge()
            if vocab_challenge:
                is_vocab_challenge = True
                from core.models import TongueRound
                # Store as VOCAB:category:word
                current_round = TongueRound(
                    f"VOCAB:{vocab_challenge['category']}:{vocab_challenge['word']}",
                    history.difficulty, 0
                )
                history.rounds.append(current_round)
                save_history(user_id)

        # Check if it's word challenge turn
        if current_round is None and history.is_word_challenge_turn():
            challenge_word = history.get_challenge_word()
            if challenge_word:
                is_word_challenge = True
                # Create a special round for word challenge
                from core.models import TongueRound
                current_round = TongueRound(f"WORD:{challenge_word['word']}", history.difficulty, 0)
                history.rounds.append(current_round)
                save_history(user_id)

        # If not any challenge, get regular sentence
        if current_round is None:
            # Generate story if needed
            if history.needs_new_story():
                story, ms = ai_provider.generate_story(
                    history.correct_words,
                    history.difficulty
                )
                history.set_story(story, history.difficulty, ms)
                save_history(user_id)

            # Get next sentence
            current_round, is_review = history.get_next_sentence()
            if not current_round:
                raise HTTPException(status_code=500, detail="No sentences available")
            save_history(user_id)

    round = current_round

    # Check for previous evaluation
    has_previous = history.last_evaluated_round is not None
    previous_eval = None
    if has_previous and history.last_evaluated_round:
        prev = history.last_evaluated_round
        previous_eval = {
            'sentence': prev.sentence,
            'translation': prev.translation,
            'score': prev.get_score(),
            'correct_translation': prev.judgement.get('correct_translation') if prev.judgement else None,
            'evaluation': prev.judgement.get('evaluation') if prev.judgement else None,
            'judge_ms': prev.judge_ms,
            'level_changed': history.last_level_changed
        }
        history.last_evaluated_round = None

    # For challenges, return just the word; for sentences, return the sentence
    sentence = round.sentence
    if is_word_challenge and sentence.startswith("WORD:"):
        sentence = sentence[5:]  # Remove prefix for display
    elif is_vocab_challenge and sentence.startswith("VOCAB:"):
        # Extract just the word from VOCAB:category:word
        parts = sentence.split(":", 2)
        sentence = parts[2] if len(parts) == 3 else sentence
    elif is_verb_challenge and sentence.startswith("VERB:"):
        sentence = sentence[5:]  # Remove prefix for display

    return NextSentenceResponse(
        sentence=sentence,
        difficulty=round.difficulty,
        story=history.current_story or "",
        sentences_remaining=len(history.story_sentences),
        progress_display=history.get_progress_display(),
        is_review=is_review,
        is_word_challenge=is_word_challenge,
        challenge_word=challenge_word,
        is_vocab_challenge=is_vocab_challenge,
        vocab_challenge=vocab_challenge,
        is_verb_challenge=is_verb_challenge,
        verb_challenge=verb_challenge,
        has_previous_evaluation=has_previous,
        previous_evaluation=previous_eval
    )


@app.post("/api/translate", response_model=TranslationResponse)
async def submit_translation(request: TranslationRequest):
    """Submit a translation for evaluation."""
    import logging
    logger = logging.getLogger(__name__)

    try:
        history = get_history(request.user_id)

        # Find the current round (last in rounds list)
        if not history.rounds:
            raise HTTPException(status_code=400, detail="No active round")

        current_round = history.rounds[-1]
        if current_round.evaluated:
            raise HTTPException(status_code=400, detail="Round already evaluated")

        # Handle different challenge types
        is_word_challenge = current_round.sentence.startswith("WORD:")
        is_vocab_challenge = current_round.sentence.startswith("VOCAB:")
        is_verb_challenge = current_round.sentence.startswith("VERB:")

        if is_verb_challenge:
            # Verb challenge: check both translation AND tense
            conjugated_form = current_round.sentence[5:]  # Remove "VERB:" prefix
            logger.info(f"Verb challenge for: {conjugated_form}")

            # Get verb info from storage (should already be there from get_next_sentence)
            stored = storage.get_verb_conjugation(conjugated_form)
            if not stored:
                # Fallback: query AI
                ai_result = ai_provider.analyze_verb_conjugation(conjugated_form)
                if ai_result:
                    storage.save_verb_conjugation(
                        conjugated_form, ai_result['base_verb'], ai_result['tense'],
                        ai_result['translation'], ai_result['person']
                    )
                    stored = ai_result

            if not stored:
                raise HTTPException(status_code=500, detail="Could not analyze verb")

            correct_translation = stored['translation']
            correct_tense = stored['tense']

            # Parse correct answers (comma-separated)
            correct_answers = [t.strip().lower() for t in correct_translation.split(',')]

            # Check translation (case-insensitive)
            student_answer = request.translation.strip().lower()
            translation_correct = student_answer in correct_answers

            # Check tense
            selected_tense = (request.selected_tense or '').lower()
            tense_correct = selected_tense == correct_tense

            # Both must be correct for full score
            if translation_correct and tense_correct:
                score = 100
                evaluation = 'Correct!'
            elif translation_correct:
                score = 50
                evaluation = f'Translation correct, but the tense is {correct_tense}, not {selected_tense}.'
            elif tense_correct:
                score = 50
                evaluation = f'Tense correct, but the translation should be: {correct_translation}'
            else:
                score = 0
                evaluation = f'The translation is "{correct_translation}" and the tense is {correct_tense}.'

            judgement = {
                'score': score,
                'correct_translation': correct_translation,
                'correct_tense': correct_tense,
                'evaluation': evaluation,
                'vocabulary_breakdown': [[conjugated_form, correct_translation, 'verb', translation_correct and tense_correct]]
            }
            judge_ms = 0

            # Verify sentence matches
            if conjugated_form != request.sentence:
                raise HTTPException(status_code=400, detail="Sentence mismatch")

        elif is_vocab_challenge:
            # Vocabulary category challenge: simple matching
            # Format: VOCAB:category:word
            parts = current_round.sentence.split(":", 2)
            if len(parts) != 3:
                raise HTTPException(status_code=400, detail="Invalid vocab challenge format")

            category = parts[1]
            word = parts[2]
            logger.info(f"Vocab challenge: category={category}, word={word}")

            from core.vocabulary import get_category_items
            items = get_category_items(category)
            correct_translation = items.get(word, '')

            # Parse correct answers (comma-separated)
            correct_answers = [t.strip().lower() for t in correct_translation.split(',')]

            # Check if translation matches (case-insensitive)
            student_answer = request.translation.strip().lower()
            is_correct = student_answer in correct_answers

            score = 100 if is_correct else 0
            judgement = {
                'score': score,
                'correct_translation': correct_translation,
                'evaluation': 'Correct!' if is_correct else f'The correct translation is: {correct_translation}',
                'vocabulary_breakdown': [[word, correct_translation, category, is_correct]]
            }
            judge_ms = 0

            # Record vocab challenge result
            history.record_vocab_result(category, word, is_correct)

            # Verify sentence matches
            if word != request.sentence:
                raise HTTPException(status_code=400, detail="Sentence mismatch")

        elif is_word_challenge:
            # Word challenge: get translation from storage or AI
            word = current_round.sentence[5:]  # Remove "WORD:" prefix
            logger.info(f"Word challenge for word: {word}")

            # First check persistent storage for translation
            stored_translation = storage.get_word_translation(word)

            if stored_translation:
                logger.info(f"Using stored translation: {stored_translation}")
                correct_translation = stored_translation['translation']
                word_type = stored_translation['type']
            else:
                # Query AI for translation and store it
                logger.info(f"Querying AI for translation of: {word}")
                ai_result = ai_provider.get_word_translation(word)
                if ai_result:
                    correct_translation = ai_result['translation']
                    word_type = ai_result['type']
                    # Save to persistent storage
                    storage.save_word_translation(word, correct_translation, word_type)
                    logger.info(f"Saved translation: {correct_translation} ({word_type})")
                else:
                    # Fallback to user's word history if AI fails
                    word_info = history.words.get(word, {})
                    correct_translation = word_info.get('translation') or ''
                    word_type = word_info.get('type') or 'unknown'
                    # Handle list translations
                    if isinstance(correct_translation, list):
                        correct_translation = ', '.join(correct_translation)

            # Parse correct answers (comma-separated)
            correct_answers = [t.strip().lower() for t in correct_translation.split(',')]

            # Check if translation matches (case-insensitive)
            student_answer = request.translation.strip().lower()
            is_correct = student_answer in correct_answers

            score = 100 if is_correct else 0
            judgement = {
                'score': score,
                'correct_translation': correct_translation,
                'evaluation': 'Correct!' if is_correct else f'The correct translation is: {correct_translation}',
                'vocabulary_breakdown': [[word, correct_translation, word_type, is_correct]]
            }
            judge_ms = 0

            # Verify sentence matches (use the word for word challenges)
            if word != request.sentence:
                raise HTTPException(status_code=400, detail="Sentence mismatch")
        else:
            # Regular sentence: AI validation
            if current_round.sentence != request.sentence:
                raise HTTPException(status_code=400, detail="Sentence mismatch")

            judgement, judge_ms = ai_provider.validate_translation(
                request.sentence,
                request.translation
            )

        current_round.translation = request.translation

        # Challenges have separate scoring, don't affect level progress
        if is_verb_challenge or is_vocab_challenge or is_word_challenge:
            current_round.judgement = judgement
            current_round.judge_ms = judge_ms
            current_round.evaluated = True
            history.total_completed += 1

            # Record challenge stats
            challenge_type = 'verb' if is_verb_challenge else ('vocab' if is_vocab_challenge else 'word')
            is_fully_correct = judgement.get('score', 0) >= 100
            history.record_challenge_result(challenge_type, is_fully_correct)

            # For word challenges, also update word tracking for learning
            if is_word_challenge:
                history.update_words(current_round, request.hint_words or [])

            save_history(request.user_id)

            return TranslationResponse(
                score=current_round.get_score(),
                correct_translation=judgement.get('correct_translation', ''),
                evaluation=judgement.get('evaluation', ''),
                vocabulary_breakdown=judgement.get('vocabulary_breakdown', []),
                judge_ms=judge_ms,
                level_changed=False,
                new_level=history.difficulty,
                change_type=None
            )

        # Regular sentences affect level progress
        level_info = history.process_evaluation(judgement, judge_ms, current_round, request.hint_words, request.hint_used)
        save_history(request.user_id)

        return TranslationResponse(
            score=current_round.get_score(),
            correct_translation=judgement.get('correct_translation', ''),
            evaluation=judgement.get('evaluation', ''),
            vocabulary_breakdown=judgement.get('vocabulary_breakdown', []),
            judge_ms=judge_ms,
            level_changed=level_info['level_changed'],
            new_level=level_info['new_level'],
            change_type=level_info['change_type']
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in submit_translation: {type(e).__name__}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal error: {type(e).__name__}: {str(e)}")


@app.post("/api/hint", response_model=HintResponse)
async def get_hint(request: HintRequest):
    """Get a hint for the current sentence."""
    history = get_history(request.user_id)

    hint = ai_provider.get_hint(request.sentence, history.correct_words)
    if not hint:
        return HintResponse(noun=None, verb=None, adjective=None)

    return HintResponse(
        noun=hint.get('noun'),
        verb=hint.get('verb'),
        adjective=hint.get('adjective')
    )


@app.get("/api/learning-words")
async def get_learning_words(user_id: str = "default"):
    """Get words that are still being learned (not yet mastered)."""
    history = get_history(user_id)
    words = history.get_learning_words()

    return {
        "total": len(words),
        "words": words
    }


@app.get("/api/mastered-words")
async def get_mastered_words(user_id: str = "default"):
    """Get mastered words (>=80% success rate and at least 2 correct)."""
    history = get_history(user_id)
    words = history.get_mastered_words()

    return {
        "total": len(words),
        "words": words
    }


@app.get("/api/stats")
async def get_api_stats():
    """Get Gemini API usage statistics."""
    return ai_provider.get_stats()


def create_app():
    """Factory function for creating the app (useful for testing)."""
    return app

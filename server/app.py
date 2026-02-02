"""FastAPI server for tongue application."""

import asyncio
import logging
import os
import random
import time
import traceback
import unicodedata
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

logger = logging.getLogger(__name__)

from core.models import History, TongueRound
from core.config import (
    DEFAULT_LANGUAGE, MIN_DIFFICULTY, MAX_DIFFICULTY,
    ADVANCE_SCORE_THRESHOLD, ADVANCE_REQUIRED_GOOD, ADVANCE_WINDOW_SIZE,
    DEMOTE_SCORE_THRESHOLD, DEMOTE_REQUIRED_POOR, STORY_SENTENCE_COUNT
)

from server.gemini_provider import GeminiProvider
from server.file_storage import FileStorage
from server.postgres_storage import PostgresStorage

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
WEB_DIR = PROJECT_ROOT / "web"


# Default accent-matters words (Spanish), used as fallback
_DEFAULT_ACCENT_WORDS = {
    'el', 'tu', 'mi', 'si', 'se', 'de', 'te', 'mas',
    'que', 'como', 'donde', 'cuando', 'cual', 'quien',
    'aun', 'solo',
}


def strip_accents(s: str) -> str:
    """Remove Unicode diacritics (accents) from a string."""
    nfkd = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in nfkd if unicodedata.category(c) != 'Mn')


def validate_script(text: str, expected_script: str) -> bool:
    """Check if text uses the expected script (cyrillic or latin).
    Returns True if the text passes validation."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return True
    cyrillic = sum(1 for c in alpha if 'CYRILLIC' in unicodedata.name(c, ''))
    if expected_script == 'cyrillic':
        return cyrillic / len(alpha) > 0.5
    if expected_script == 'latin':
        return cyrillic / len(alpha) < 0.5
    return True


def accent_lenient_match(student: str, correct: str, target_is_spanish: bool = True,
                         accent_words: set = None) -> bool:
    """Check if student matches correct answer, forgiving accent differences.

    For target language, exact accents are still required on words where
    the accent changes meaning.
    For English targets, accents are always stripped freely.
    """
    stripped_student = strip_accents(student.lower())
    stripped_correct = strip_accents(correct.lower())
    if stripped_student != stripped_correct:
        return False
    # If targeting English, accent differences never matter
    if not target_is_spanish:
        return True
    # For target language, reject if the word is in the accent-matters list
    accent_set = accent_words if accent_words is not None else _DEFAULT_ACCENT_WORDS
    student_words = student.lower().split()
    correct_words_list = correct.lower().split()
    if len(student_words) != len(correct_words_list):
        return True  # length mismatch handled by stripped comparison above
    for sw, cw in zip(student_words, correct_words_list):
        stripped_cw = strip_accents(cw)
        if stripped_cw in accent_set and sw != cw:
            return False
    return True


# Language info cache
_language_cache: dict[str, dict] = {}


def get_language_info(lang_code: str) -> dict | None:
    """Get language info from DB with in-memory cache."""
    if lang_code in _language_cache:
        return _language_cache[lang_code]
    if storage:
        info = storage.get_language(lang_code)
        if info:
            _language_cache[lang_code] = info
            return info
    return None


def get_user_language_info(user_id: str) -> dict:
    """Get language info for user's current language."""
    history = get_history(user_id)
    info = get_language_info(history.language)
    if info:
        return info
    # Fallback to Spanish defaults
    return {
        'code': 'es', 'name': 'Español', 'script': 'latin',
        'english_name': 'Spanish',
        'tenses': ['present', 'preterite', 'imperfect', 'future', 'conditional', 'subjunctive'],
        'accent_words': list(_DEFAULT_ACCENT_WORDS)
    }


def get_accent_words_set(language_info: dict) -> set:
    """Get accent words as a set from language info."""
    words = language_info.get('accent_words', [])
    if isinstance(words, list):
        return set(words)
    return set()


# Pydantic models for API
class TranslationRequest(BaseModel):
    sentence: str
    translation: str
    user_id: str = "default"
    hint_used: bool = False
    hint_words: list[str] = []
    selected_tense: Optional[str] = None  # For verb challenges
    translations: list[str] = []  # For multi-word vocab challenges (4 answers)


class HintRequest(BaseModel):
    sentence: str
    user_id: str = "default"
    partial_translation: str = ""


class CreateUserRequest(BaseModel):
    pin: str
    language: str = 'es'


class LoginRequest(BaseModel):
    pin: str


class ErrorLogRequest(BaseModel):
    error: str
    endpoint: str = ""
    user_id: str = ""
    status: int = 0
    context: str = ""


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
    word_results: Optional[list] = None  # Per-word results for multi-word vocab challenges


class HintResponse(BaseModel):
    noun: Optional[list]
    verb: Optional[list]
    adjective: Optional[list]


class VerbHintRequest(BaseModel):
    sentence: str
    user_id: str = "default"


class VerbHintResponse(BaseModel):
    rules: str
    tense: str
    base_verb: str


class StatusResponse(BaseModel):
    language: str
    language_code: str = 'es'
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
    practice_time_seconds: int
    practice_time_display: str  # Human-readable format like "2h 15m"
    practice_times: dict = {}  # Per language+direction breakdown, e.g. {"es:normal": 1380}
    direction: str = 'normal'  # 'normal' (ES→EN) or 'reverse' (EN→ES)
    tenses: list[str] = []  # Available tenses for current language


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
    is_synonym_challenge: bool = False
    synonym_challenge: Optional[dict] = None  # {word, challenge_type (SYN/ANT), type}
    has_previous_evaluation: bool
    previous_evaluation: Optional[dict]
    direction: str = 'normal'


# Global state (in production, use proper DI)
storage: FileStorage = None
ai_provider: GeminiProvider = None  # Fast model for validation/hints
story_provider: GeminiProvider = None  # Pro model for story generation
user_histories: dict[str, History] = {}

# Background story generation
pending_stories: dict[str, dict] = {}  # user_id -> {story, difficulty, ms, task}
story_generation_locks: dict[str, asyncio.Lock] = {}  # Prevent duplicate generations

# Session tracking
user_sessions: dict[str, str] = {}  # user_id -> session_id

# Practice time tracking - when sentences were served to users
user_served_times: dict[str, float] = {}  # user_id -> timestamp


def get_session_id(user_id: str) -> str:
    """Get or create a session ID for a user."""
    if user_id not in user_sessions:
        user_sessions[user_id] = str(uuid.uuid4())[:8]
    return user_sessions[user_id]


def new_session(user_id: str) -> str:
    """Create a new session for a user."""
    user_sessions[user_id] = str(uuid.uuid4())[:8]
    return user_sessions[user_id]


def record_served_time(user_id: str) -> None:
    """Record when a sentence/challenge was served to a user."""
    user_served_times[user_id] = time.time()


def get_practice_delta(user_id: str) -> float | None:
    """Get practice time delta since sentence was served.
    Returns delta in seconds, or None if no served time recorded."""
    served_time = user_served_times.get(user_id)
    if served_time is not None:
        delta = time.time() - served_time
        del user_served_times[user_id]  # Clear to prevent double-counting
        return delta
    return None


def format_practice_time(seconds: int) -> str:
    """Format seconds as human-readable time."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 600:
        return f"{seconds // 60}m {seconds % 60}s"
    elif seconds < 3600:
        return f"{seconds // 60}m"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"


def log_event(event: str, user_id: str, **data) -> None:
    """Log an event to the database."""
    if storage and hasattr(storage, 'log_event'):
        history = user_histories.get(user_id)
        difficulty = history.difficulty if history else None
        session_id = get_session_id(user_id)
        # Extract AI-related fields so they go into dedicated columns
        ai_used = data.pop('ai_used', False)
        model_name = data.pop('model_name', None)
        model_tokens = data.pop('model_tokens', None)
        model_ms = data.pop('model_ms', None)
        storage.log_event(event, user_id, session_id, difficulty,
                          ai_used=ai_used, model_name=model_name,
                          model_tokens=model_tokens, model_ms=model_ms,
                          **data)


app = FastAPI(title="Tongue API", description="Multi-language translation practice API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://tongue.pr3da.com",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/app")
async def serve_app():
    """Serve the web application."""
    return FileResponse(WEB_DIR / "index.html")


@app.get("/logs")
async def serve_logs():
    """Serve the event logs page."""
    return FileResponse(WEB_DIR / "logs.html")


@app.get("/perf")
async def serve_perf():
    """Serve the performance stats page."""
    return FileResponse(WEB_DIR / "perf.html")


def get_history(user_id: str = "default") -> History:
    """Get or create history for a user.

    Only creates a new empty History for users that genuinely don't exist yet
    (just created via create_user). Database errors will propagate as exceptions
    to prevent overwriting existing data with an empty history.
    """
    user_id = user_id.lower()
    if user_id not in user_histories:
        state = storage.load_state(user_id)  # Raises on DB error
        if state:
            user_histories[user_id] = History.from_dict(state)
        else:
            # Only safe because create_user already verified this is a new user
            user_histories[user_id] = History()
    return user_histories[user_id]


def save_history(user_id: str = "default") -> None:
    """Save history for a user."""
    user_id = user_id.lower()
    if user_id in user_histories:
        storage.save_state(user_histories[user_id].to_dict(), user_id)


async def generate_story_background(user_id: str, correct_words: list, difficulty: int, direction: str = 'normal', language_info: dict = None) -> None:
    """Generate a story in the background for a user."""
    # Get or create lock for this user
    if user_id not in story_generation_locks:
        story_generation_locks[user_id] = asyncio.Lock()

    async with story_generation_locks[user_id]:
        # Check if we already have a pending story at this difficulty and direction
        if user_id in pending_stories:
            pending = pending_stories[user_id]
            if (pending.get('difficulty') == difficulty and
                    pending.get('direction', 'normal') == direction and
                    pending.get('story')):
                logger.info(f"Story already pre-generated for {user_id} at difficulty {difficulty}")
                return

        logger.info(f"Background generating story for {user_id} at difficulty {difficulty} direction={direction}")
        try:
            # Run in executor to not block the event loop
            loop = asyncio.get_event_loop()
            li = language_info
            story, ms = await loop.run_in_executor(
                None,
                lambda: story_provider.generate_story(correct_words, difficulty, direction, language_info=li)
            )
            story_ai = story_provider.get_last_call_info()
            pending_stories[user_id] = {
                'story': story,
                'difficulty': difficulty,
                'direction': direction,
                'ms': ms,
                'model_name': story_ai.get('model_name'),
                'model_tokens': story_ai.get('model_tokens'),
                'model_ms': story_ai.get('model_ms')
            }
            logger.info(f"Background story ready for {user_id}: {len(story)} chars, {ms}ms")
        except Exception as e:
            logger.error(f"Background story generation failed for {user_id}: {e}")
            # Clear any partial state
            pending_stories.pop(user_id, None)


def get_pending_story(user_id: str, difficulty: int, direction: str = 'normal') -> tuple[str, int, dict] | None:
    """Get a pre-generated story if available and at the right difficulty and direction."""
    if user_id in pending_stories:
        pending = pending_stories[user_id]
        if (pending.get('difficulty') == difficulty and
                pending.get('direction', 'normal') == direction and
                pending.get('story')):
            story = pending['story']
            ms = pending['ms']
            ai_info = {
                'model_name': pending.get('model_name'),
                'model_tokens': pending.get('model_tokens'),
                'model_ms': pending.get('model_ms')
            }
            # Clear the pending story
            del pending_stories[user_id]
            logger.info(f"Using pre-generated story for {user_id}")
            return (story, ms, ai_info)
    return None


def trigger_background_story(user_id: str, history: History) -> None:
    """Trigger background story generation for a user."""
    # Only trigger if user might need a new story soon
    # (less than 3 sentences remaining or no story)
    if len(history.story_sentences) < 3 or history.needs_new_story():
        lang_info = get_user_language_info(user_id)
        asyncio.create_task(
            generate_story_background(user_id, history.correct_words, history.difficulty, history.direction, language_info=lang_info)
        )


@app.on_event("startup")
async def startup():
    """Initialize storage and AI providers on startup."""
    global storage, ai_provider, story_provider

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

    # Fast model for validation, hints, word/verb analysis
    ai_provider = GeminiProvider(api_key, model_name='gemini-2.0-flash', storage=storage)
    # Pro model for higher quality story generation
    story_provider = GeminiProvider(api_key, model_name='gemini-2.5-pro', storage=storage)
    print("AI providers initialized: gemini-2.0-flash (validation), gemini-2.5-pro (stories)")

    # Initialize vocabulary storage and seed DB for all active languages
    from core.vocabulary import init_storage as vocab_init_storage, get_seed_data
    vocab_init_storage(storage)
    languages = storage.get_languages()
    for lang in languages:
        seed_data = get_seed_data(lang['code'])
        if seed_data:
            storage.seed_vocabulary(seed_data)
    print(f"Vocabulary storage initialized and seeded for {len(languages)} languages")


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
    user_id = user_id.lower()
    exists = storage.user_exists(user_id)
    return {"exists": exists}


@app.post("/api/users/{user_id}")
async def create_user(user_id: str, request: CreateUserRequest):
    """Create a new user with a PIN. Returns error if user already exists."""
    user_id = user_id.lower()
    if storage.user_exists(user_id):
        return {"success": False, "error": "User already exists"}

    # Validate PIN is 4 digits
    if not request.pin or len(request.pin) != 4 or not request.pin.isdigit():
        return {"success": False, "error": "PIN must be exactly 4 digits"}

    # Create empty history for new user with selected language
    history = History()
    lang_code = request.language or DEFAULT_LANGUAGE
    # Validate language exists
    lang_info = get_language_info(lang_code)
    if lang_info:
        history.language = lang_code
    else:
        history.language = DEFAULT_LANGUAGE
    user_histories[user_id] = history
    save_history(user_id)

    # Save the PIN
    storage.save_pin(user_id, request.pin)

    # Log user creation and start new session
    session_id = new_session(user_id)
    log_event('user.create', user_id, language=history.language)

    return {"success": True, "user_id": user_id}


@app.post("/api/users/{user_id}/login")
async def login_user(user_id: str, request: LoginRequest):
    """Login an existing user with PIN verification."""
    user_id = user_id.lower()
    if not storage.user_exists(user_id):
        return {"success": False, "error": "User not found"}

    # Check if user has a PIN set
    existing_pin = storage.get_pin_hash(user_id)

    if not existing_pin:
        # Legacy user without PIN - set their PIN now
        storage.save_pin(user_id, request.pin)
        session_id = new_session(user_id)
        log_event('user.login', user_id, is_legacy=True)
        return {"success": True, "user_id": user_id}

    # Verify PIN
    if not storage.verify_pin(user_id, request.pin):
        return {
            "success": False,
            "error": "User name already exists. Please enter the correct PIN or choose a different name to start a new game."
        }

    # Start new session on successful login
    session_id = new_session(user_id)
    log_event('user.login', user_id)

    return {"success": True, "user_id": user_id}


@app.get("/api/languages")
async def list_languages():
    """List all active languages."""
    languages = storage.get_languages()
    return {"languages": languages}


@app.post("/api/switch-language")
async def switch_language(user_id: str = "default", language: str = "es"):
    """Switch user's active language."""
    user_id = user_id.lower()
    history = get_history(user_id)
    old_language = history.language

    # Validate language exists
    lang_info = get_language_info(language)
    if not lang_info:
        return {"success": False, "error": f"Unknown language: {language}"}

    history.switch_language(language)
    save_history(user_id)

    # Clear pending stories (wrong language)
    pending_stories.pop(user_id, None)

    # Trigger background story generation for new language
    trigger_background_story(user_id, history)

    log_event('language.switch', user_id,
              old_language=old_language,
              new_language=language)

    return {"success": True, "language": language}


@app.get("/api/status", response_model=StatusResponse)
async def get_status(user_id: str = "default"):
    """Get user status and progress."""
    try:
        history = get_history(user_id)
        lang_info = get_user_language_info(user_id)

        return StatusResponse(
            language=lang_info['english_name'],
            language_code=lang_info['code'],
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
            challenge_stats_display=history.get_challenge_stats_display(),
            practice_time_seconds=int(history.current_practice_time_seconds),
            practice_time_display=format_practice_time(int(history.current_practice_time_seconds)),
            practice_times={k: int(v) for k, v in history.practice_times.items()},
            direction=history.direction,
            tenses=lang_info.get('tenses', [])
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_status: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        log_event('error.server', user_id,
                  endpoint='/api/status',
                  error_type=type(e).__name__,
                  error_message=str(e),
                  traceback=traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


@app.get("/api/story", response_model=StoryResponse)
async def get_story(user_id: str = "default", force_new: bool = False):
    """Get current story or generate a new one."""
    history = get_history(user_id)

    if force_new or history.needs_new_story():
        lang_info = get_user_language_info(user_id)
        # Try to use pre-generated story first
        pending = get_pending_story(user_id, history.difficulty, history.direction)
        from_cache = pending is not None
        if pending:
            story, ms, story_ai = pending
        else:
            # Fall back to synchronous generation with story_provider (pro model)
            loop = asyncio.get_event_loop()
            direction = history.direction
            li = lang_info
            story, ms = await loop.run_in_executor(
                None,
                lambda: story_provider.generate_story(history.correct_words, history.difficulty, direction, language_info=li)
            )
            story_ai = story_provider.get_last_call_info()
        history.set_story(story, history.difficulty, ms)
        save_history(user_id)

        # Log story generation
        log_event('story.generate', user_id,
                  from_cache=from_cache,
                  ms=ms,
                  sentence_count=len(history.story_sentences),
                  ai_used=True,
                  model_name=story_ai.get('model_name'),
                  model_tokens=story_ai.get('model_tokens'),
                  model_ms=story_ai.get('model_ms'))

        # Trigger background generation for next story
        trigger_background_story(user_id, history)

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
    try:
        return await _get_next_sentence_inner(user_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_next_sentence: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        log_event('error.server', user_id,
                  endpoint='/api/next',
                  error_type=type(e).__name__,
                  error_message=str(e),
                  traceback=traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


async def _get_next_sentence_inner(user_id: str = "default"):
    start_time = time.time()
    history = get_history(user_id)
    lang_code = history.language

    # Check if there's an existing unevaluated round (e.g., after page refresh)
    current_round = None
    is_review = False
    is_word_challenge = False
    is_vocab_challenge = False
    is_verb_challenge = False
    is_synonym_challenge = False
    challenge_word = None
    vocab_challenge = None
    verb_challenge = None
    synonym_challenge = None

    if history.rounds:
        last_round = history.rounds[-1]
        if not last_round.evaluated:
            current_round = last_round
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
            # Check if this was a multi-word vocab challenge
            elif last_round.sentence.startswith("VOCAB4:") or last_round.sentence.startswith("VOCAB4R:"):
                is_vocab_challenge = True
                is_reverse = last_round.sentence.startswith("VOCAB4R:")
                prefix = "VOCAB4R:" if is_reverse else "VOCAB4:"
                rest = last_round.sentence[len(prefix):]
                parts = rest.split(":", 1)
                if len(parts) == 2:
                    from core.vocabulary import get_category_name
                    category = parts[0]
                    english_keys = parts[1].split(",")
                    # Look up each english key to get word/alternatives
                    words = []
                    for eng_key in english_keys:
                        item = storage.get_vocab_item_by_english(category, eng_key, language=lang_code)
                        if item:
                            words.append({'word': item['word'], 'translation': item['alternatives'], 'english': item['english']})
                        else:
                            words.append({'word': eng_key, 'translation': eng_key, 'english': eng_key})
                    vocab_challenge = {
                        'words': words,
                        'category': category,
                        'category_name': get_category_name(category),
                        'is_multi': True,
                        'is_reverse': is_reverse
                    }
            # Check if this was a single-word vocab challenge (sentence starts with "VOCAB:" or "VOCABR:")
            elif last_round.sentence.startswith("VOCAB:") or last_round.sentence.startswith("VOCABR:"):
                is_vocab_challenge = True
                is_reverse = last_round.sentence.startswith("VOCABR:")
                # Format: VOCAB:category:english_key or VOCABR:category:english_key
                parts = last_round.sentence.split(":", 2)
                if len(parts) == 3:
                    from core.vocabulary import get_category_name
                    category = parts[1]
                    english_key = parts[2]
                    item = storage.get_vocab_item_by_english(category, english_key, language=lang_code)
                    if item:
                        vocab_challenge = {
                            'word': item['word'],
                            'translation': item['alternatives'],
                            'english': item['english'],
                            'category': category,
                            'category_name': get_category_name(category),
                            'is_reverse': is_reverse
                        }
                    else:
                        vocab_challenge = {
                            'word': english_key,
                            'translation': english_key,
                            'english': english_key,
                            'category': category,
                            'category_name': get_category_name(category),
                            'is_reverse': is_reverse
                        }
            # Check if this was a verb challenge (sentence starts with "VERB:")
            elif last_round.sentence.startswith("VERB:"):
                is_verb_challenge = True
                conjugated_form = last_round.sentence[5:]  # Remove "VERB:" prefix
                # Get verb info from storage
                stored = storage.get_verb_conjugation(conjugated_form, language=lang_code)
                if stored:
                    verb_challenge = {
                        'conjugated_form': conjugated_form,
                        **stored
                    }
            # Check if this was a synonym/antonym challenge
            elif last_round.sentence.startswith("SYN:") or last_round.sentence.startswith("ANT:"):
                is_synonym_challenge = True
                challenge_type = "SYN" if last_round.sentence.startswith("SYN:") else "ANT"
                syn_word = last_round.sentence[4:]  # Remove "SYN:" or "ANT:" prefix
                word_info = history.words.get(syn_word, {})
                synonym_challenge = {
                    'word': syn_word,
                    'challenge_type': challenge_type,
                    'type': word_info.get('type', 'unknown')
                }
            else:
                # Not a challenge - check if it's a review sentence (generate_ms=0)
                is_review = last_round.generate_ms == 0

    # Only get a new sentence/challenge if there's no current unevaluated round
    lang_code = history.language
    lang_info = get_user_language_info(user_id)
    if current_round is None:
        # Check if it's verb challenge turn (every 7th turn)
        if history.is_verb_challenge_turn():
            verb_word = history.get_verb_for_challenge()
            if verb_word:
                # Check storage for existing conjugation, or query AI
                stored = storage.get_verb_conjugation(verb_word, language=lang_code)
                if stored:
                    verb_challenge = {'conjugated_form': verb_word, **stored}
                else:
                    # Query AI for verb conjugation
                    ai_result = ai_provider.analyze_verb_conjugation(verb_word, language_info=lang_info)
                    if ai_result:
                        storage.save_verb_conjugation(
                            verb_word, ai_result['base_verb'], ai_result['tense'],
                            ai_result['translation'], ai_result['person'],
                            language=lang_code
                        )
                        verb_challenge = {'conjugated_form': verb_word, **ai_result}

                if verb_challenge:
                    is_verb_challenge = True
                    from core.models import TongueRound
                    current_round = TongueRound(f"VERB:{verb_word}", history.difficulty, 0)
                    history.rounds.append(current_round)
                    save_history(user_id)

        # Check if it's synonym/antonym challenge turn (every 11th turn)
        if current_round is None and history.is_synonym_challenge_turn():
            syn_word_info = history.get_word_for_synonym_challenge()
            if syn_word_info:
                syn_word = syn_word_info['word']
                # Check cache first
                cached = storage.get_synonym_antonym(syn_word, language=lang_code)
                if not cached:
                    # Generate via AI and cache
                    ai_result = ai_provider.generate_synonym_antonym(syn_word, language_info=lang_info)
                    if ai_result:
                        storage.save_synonym_antonym(
                            syn_word, lang_code,
                            ai_result.get('synonym'), ai_result.get('antonym')
                        )
                        cached = ai_result

                if cached:
                    # Randomly pick SYN or ANT
                    has_synonym = cached.get('synonym') is not None
                    has_antonym = cached.get('antonym') is not None

                    if has_synonym and has_antonym:
                        challenge_type = random.choice(['SYN', 'ANT'])
                    elif has_synonym:
                        challenge_type = 'SYN'
                    elif has_antonym:
                        challenge_type = 'ANT'
                    else:
                        challenge_type = None

                    if challenge_type:
                        is_synonym_challenge = True
                        synonym_challenge = {
                            'word': syn_word,
                            'challenge_type': challenge_type,
                            'type': syn_word_info['type']
                        }
                        from core.models import TongueRound
                        current_round = TongueRound(f"{challenge_type}:{syn_word}", history.difficulty, 0)
                        history.rounds.append(current_round)
                        save_history(user_id)

        # Check if it's vocab challenge turn (every 4th turn, offset by 2)
        if current_round is None and history.is_vocab_challenge_turn():
            vocab_challenge = history.get_vocab_challenge()
            if vocab_challenge:
                is_vocab_challenge = True
                from core.models import TongueRound
                if vocab_challenge.get('is_multi'):
                    # Multi-word: store as VOCAB4:category:eng1,eng2,eng3,eng4
                    # or VOCAB4R:category:eng1,eng2,eng3,eng4
                    words_str = ','.join(w['english'] for w in vocab_challenge['words'])
                    prefix = 'VOCAB4R' if vocab_challenge.get('is_reverse') else 'VOCAB4'
                    current_round = TongueRound(
                        f"{prefix}:{vocab_challenge['category']}:{words_str}",
                        history.difficulty, 0
                    )
                else:
                    # Single-word: store as VOCAB:category:english_key
                    # or VOCABR:category:english_key for reverse (en->es)
                    prefix = 'VOCABR' if vocab_challenge.get('is_reverse') else 'VOCAB'
                    current_round = TongueRound(
                        f"{prefix}:{vocab_challenge['category']}:{vocab_challenge['english']}",
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
                # Try to use pre-generated story first
                pending = get_pending_story(user_id, history.difficulty, history.direction)
                from_cache = pending is not None
                if pending:
                    story, ms, story_ai = pending
                else:
                    # Fall back to synchronous generation with story_provider (pro model)
                    loop = asyncio.get_event_loop()
                    direction = history.direction
                    li = lang_info
                    story, ms = await loop.run_in_executor(
                        None,
                        lambda: story_provider.generate_story(history.correct_words, history.difficulty, direction, language_info=li)
                    )
                    story_ai = story_provider.get_last_call_info()
                history.set_story(story, history.difficulty, ms)
                save_history(user_id)

                # Log story generation
                log_event('story.generate', user_id,
                          from_cache=from_cache,
                          ms=ms,
                          sentence_count=len(history.story_sentences),
                          ai_used=True,
                          model_name=story_ai.get('model_name'),
                          model_tokens=story_ai.get('model_tokens'),
                          model_ms=story_ai.get('model_ms'))

            # Get next sentence
            current_round, is_review = history.get_next_sentence()
            if not current_round:
                raise HTTPException(status_code=500, detail="No sentences available")
            save_history(user_id)

            # Trigger background generation if running low on sentences
            trigger_background_story(user_id, history)

    round = current_round

    # Check for previous evaluation
    has_previous = history.last_evaluated_round is not None
    previous_eval = None
    if has_previous and history.last_evaluated_round:
        prev = history.last_evaluated_round
        prev_sentence = prev.sentence

        # Determine challenge type and clean up sentence for display
        prev_challenge_type = None
        if prev_sentence.startswith("WORD:"):
            prev_challenge_type = 'word'
            word = prev_sentence[5:]  # Remove WORD: prefix
            # In reverse mode user was shown English, so display English
            if history.direction == 'reverse':
                word_info = history.words.get(word, {})
                trans = word_info.get('translation', '')
                if isinstance(trans, list):
                    trans = ', '.join(trans)
                if trans and trans.lower().strip() != word.lower().strip():
                    prev_sentence = trans
                else:
                    # Fallback: try persistent storage
                    stored = storage.get_word_translation(word, language=lang_code)
                    prev_sentence = stored['translation'] if stored else word
            else:
                prev_sentence = word
        elif prev_sentence.startswith("VOCAB4R:") or prev_sentence.startswith("VOCAB4:"):
            prev_challenge_type = 'vocab'
            is_rev_vocab = prev_sentence.startswith("VOCAB4R:")
            prefix = "VOCAB4R:" if is_rev_vocab else "VOCAB4:"
            rest = prev_sentence[len(prefix):]
            parts = rest.split(":", 1)
            if len(parts) == 2:
                eng_keys = parts[1].split(",")
                if is_rev_vocab:
                    # Reverse: user saw English words
                    prev_sentence = ', '.join(eng_keys)
                else:
                    # Normal: user saw target language words
                    target_words = []
                    for ek in eng_keys:
                        item = storage.get_vocab_item_by_english(parts[0], ek, language=lang_code)
                        target_words.append(item['word'] if item else ek)
                    prev_sentence = ', '.join(target_words)
            else:
                prev_sentence = prev_sentence
        elif prev_sentence.startswith("VOCABR:") or prev_sentence.startswith("VOCAB:"):
            prev_challenge_type = 'vocab'
            is_reverse_vocab = prev_sentence.startswith("VOCABR:")
            parts = prev_sentence.split(":", 2)
            if len(parts) == 3:
                if is_reverse_vocab:
                    # Reverse mode: display the English key
                    prev_sentence = parts[2]
                else:
                    # Normal mode: look up target language word from english key for display
                    item = storage.get_vocab_item_by_english(parts[1], parts[2], language=lang_code)
                    prev_sentence = item['word'] if item else parts[2]
            else:
                prev_sentence = prev_sentence
        elif prev_sentence.startswith("VERB:"):
            prev_challenge_type = 'verb'
            conjugated = prev_sentence[5:]  # Remove VERB: prefix
            # In reverse mode user was shown English, so display English
            if history.direction == 'reverse':
                stored = storage.get_verb_conjugation(conjugated, language=lang_code)
                prev_sentence = stored.get('translation', conjugated) if stored else conjugated
            else:
                prev_sentence = conjugated
        elif prev_sentence.startswith("SYN:") or prev_sentence.startswith("ANT:"):
            prev_challenge_type = 'synonym'
            prev_sentence = prev_sentence[4:]  # Remove SYN: or ANT: prefix

        # Determine challenge direction for display
        prev_challenge_direction = None
        if prev_challenge_type:
            lang_upper = lang_code.upper()
            if prev_challenge_type == 'synonym':
                # Synonym challenges are always in the target language
                prev_challenge_direction = f"{lang_upper}"
            elif prev_challenge_type == 'vocab':
                # Vocab challenges have per-challenge direction via prefix
                is_rev = prev.sentence.startswith("VOCABR:") or prev.sentence.startswith("VOCAB4R:")
                prev_challenge_direction = f"EN → {lang_upper}" if is_rev else f"{lang_upper} → EN"
            else:
                # Word and verb challenges use the session direction
                is_rev = history.direction == 'reverse'
                prev_challenge_direction = f"EN → {lang_upper}" if is_rev else f"{lang_upper} → EN"

        previous_eval = {
            'sentence': prev_sentence,
            'translation': prev.translation,
            'score': prev.get_score(),
            'correct_translation': prev.judgement.get('correct_translation') if prev.judgement else None,
            'evaluation': prev.judgement.get('evaluation') if prev.judgement else None,
            'judge_ms': prev.judge_ms,
            'level_changed': history.last_level_changed,
            'challenge_type': prev_challenge_type,
            'challenge_direction': prev_challenge_direction
        }

    # For challenges, return just the word; for sentences, return the sentence
    sentence = round.sentence
    if is_word_challenge and sentence.startswith("WORD:"):
        sentence = sentence[5:]  # Remove prefix for display
        # Reverse mode: show English translation, user types target language word
        if history.direction == 'reverse' and challenge_word:
            trans = challenge_word.get('translation', sentence)
            if isinstance(trans, list):
                trans = ', '.join(trans)
            # Guard: if translation is empty or same as the word (bad data from AI),
            # keep showing the original word — the challenge will still work in forward mode
            original_word = sentence
            if trans and trans.lower().strip() != original_word.lower().strip():
                sentence = trans
    elif is_vocab_challenge and (sentence.startswith("VOCAB4:") or sentence.startswith("VOCAB4R:")):
        # Multi-word: the vocab_challenge dict has words with Spanish 'word' fields
        # sentence field isn't displayed for multi-word (UI hides it)
        prefix = "VOCAB4R:" if sentence.startswith("VOCAB4R:") else "VOCAB4:"
        rest = sentence[len(prefix):]
        parts = rest.split(":", 1)
        sentence = parts[1] if len(parts) == 2 else sentence
    elif is_vocab_challenge and (sentence.startswith("VOCAB:") or sentence.startswith("VOCABR:")):
        # Single-word vocab: display Spanish word (es->en) or English key (en->es)
        is_reverse = sentence.startswith("VOCABR:")
        if vocab_challenge:
            sentence = vocab_challenge['english'] if is_reverse else vocab_challenge['word']
        else:
            parts = sentence.split(":", 2)
            sentence = parts[2] if len(parts) == 3 else sentence
    elif is_verb_challenge and sentence.startswith("VERB:"):
        sentence = sentence[5:]  # Remove prefix for display
        # Reverse mode: show English translation, user types Spanish conjugated form
        if history.direction == 'reverse' and verb_challenge:
            sentence = verb_challenge.get('translation', sentence)
    elif is_synonym_challenge and (sentence.startswith("SYN:") or sentence.startswith("ANT:")):
        sentence = sentence[4:]  # Remove SYN: or ANT: prefix for display

    # Record when this sentence/challenge was served for practice time tracking
    record_served_time(user_id)

    # Log sentence or challenge served
    ms = int((time.time() - start_time) * 1000)
    if is_word_challenge:
        log_event('challenge.served', user_id,
                  challenge_type='word',
                  word=sentence,
                  ms=ms)
    elif is_vocab_challenge:
        log_event('challenge.served', user_id,
                  challenge_type='vocab',
                  word=sentence,
                  category=vocab_challenge.get('category') if vocab_challenge else None,
                  ms=ms)
    elif is_synonym_challenge:
        log_event('challenge.served', user_id,
                  challenge_type='synonym',
                  word=sentence,
                  syn_type=synonym_challenge.get('challenge_type') if synonym_challenge else None,
                  ms=ms)
    elif is_verb_challenge:
        log_event('challenge.served', user_id,
                  challenge_type='verb',
                  word=sentence,
                  tense=verb_challenge.get('tense') if verb_challenge else None,
                  ms=ms)
    else:
        log_event('sentence.served', user_id,
                  sentence=sentence,
                  is_review=is_review,
                  sentences_remaining=len(history.story_sentences),
                  ms=ms)

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
        is_synonym_challenge=is_synonym_challenge,
        synonym_challenge=synonym_challenge,
        has_previous_evaluation=has_previous,
        previous_evaluation=previous_eval,
        direction=history.direction
    )


@app.post("/api/translate", response_model=TranslationResponse)
async def submit_translation(request: TranslationRequest):
    """Submit a translation for evaluation."""
    import logging
    logger = logging.getLogger(__name__)
    start_time = time.time()

    try:
        history = get_history(request.user_id)
        lang_code = history.language
        lang_info = get_user_language_info(request.user_id)
        accent_set = get_accent_words_set(lang_info)
        script = lang_info.get('script', 'latin')

        # Find the current round (last in rounds list)
        if not history.rounds:
            raise HTTPException(status_code=400, detail="No active round")

        current_round = history.rounds[-1]
        if current_round.evaluated:
            raise HTTPException(status_code=400, detail="Round already evaluated")

        logger.info(f"Submit: round_sentence={current_round.sentence!r}, request_sentence={request.sentence!r}, direction={history.direction}")

        # Handle different challenge types
        is_word_challenge = current_round.sentence.startswith("WORD:")
        is_multi_vocab = (current_round.sentence.startswith("VOCAB4:") or
                          current_round.sentence.startswith("VOCAB4R:"))
        is_vocab_challenge = current_round.sentence.startswith("VOCAB:") or current_round.sentence.startswith("VOCABR:") or is_multi_vocab
        is_verb_challenge = current_round.sentence.startswith("VERB:")
        is_synonym_challenge = (current_round.sentence.startswith("SYN:") or
                                current_round.sentence.startswith("ANT:"))

        # Calculate and record practice time
        practice_delta = get_practice_delta(request.user_id)
        if practice_delta is not None:
            practice_recorded = history.record_practice_time(practice_delta)
            # Log practice time event
            challenge_type_str = 'verb' if is_verb_challenge else ('synonym' if is_synonym_challenge else ('vocab' if is_vocab_challenge else ('word' if is_word_challenge else 'sentence')))
            log_event('practice_time.delta', request.user_id,
                      delta_seconds=round(practice_delta, 2),
                      task_type=challenge_type_str,
                      recorded=practice_recorded,
                      ms=int(practice_delta * 1000))

        if is_synonym_challenge:
            # Synonym/Antonym challenge
            challenge_type = "SYN" if current_round.sentence.startswith("SYN:") else "ANT"
            syn_word = current_round.sentence[4:]  # Remove SYN: or ANT: prefix
            logger.info(f"Synonym challenge for: {syn_word} (type={challenge_type})")

            # Get cached synonym/antonym — regenerate via AI if cache was lost
            # (e.g., server restart with file storage, or DB connection hiccup)
            cached = storage.get_synonym_antonym(syn_word, language=lang_code)
            if not cached:
                logger.warning(f"Synonym/antonym cache miss for '{syn_word}', regenerating via AI")
                ai_result = ai_provider.generate_synonym_antonym(syn_word, language_info=lang_info)
                if ai_result:
                    storage.save_synonym_antonym(
                        syn_word, lang_code,
                        ai_result.get('synonym'), ai_result.get('antonym')
                    )
                    cached = ai_result
            if not cached:
                raise HTTPException(status_code=500, detail="Synonym/antonym data not found")

            expected = cached.get('synonym') if challenge_type == 'SYN' else cached.get('antonym')
            if not expected:
                raise HTTPException(status_code=500, detail=f"No {challenge_type.lower()} available for this word")

            student_answer = request.translation.strip()
            type_label = 'Synonym' if challenge_type == 'SYN' else 'Antonym'

            # Check if answer matches cached value (case-insensitive, accent-lenient)
            is_correct = student_answer.lower() == expected.lower()
            if not is_correct:
                is_correct = accent_lenient_match(student_answer, expected, target_is_spanish=True, accent_words=accent_set)

            # If doesn't match cached, ask AI to validate (user might provide a different valid answer)
            if not is_correct and student_answer:
                ai_check = ai_provider.validate_synonym_antonym(
                    syn_word, challenge_type, expected, student_answer,
                    language_info=lang_info
                )
                is_correct = ai_check.get('correct', False)

            score = 100 if is_correct else 0
            judgement = {
                'score': score,
                'correct_translation': expected,
                'evaluation': f'Correct!' if is_correct else f'The {type_label.lower()} of "{syn_word}" is: {expected}',
                'vocabulary_breakdown': [[syn_word, expected, type_label.lower(), is_correct]]
            }
            judge_ms = 0

        elif is_verb_challenge:
            # Verb challenge: check both translation AND tense
            conjugated_form = current_round.sentence[5:]  # Remove "VERB:" prefix
            logger.info(f"Verb challenge for: {conjugated_form}")

            # Get verb info from storage (should already be there from get_next_sentence)
            stored = storage.get_verb_conjugation(conjugated_form, language=lang_code)
            if not stored:
                # Fallback: query AI
                ai_result = ai_provider.analyze_verb_conjugation(conjugated_form, language_info=lang_info)
                if ai_result:
                    storage.save_verb_conjugation(
                        conjugated_form, ai_result['base_verb'], ai_result['tense'],
                        ai_result['translation'], ai_result['person'], language=lang_code
                    )
                    stored = ai_result

            if not stored:
                raise HTTPException(status_code=500, detail="Could not analyze verb")

            english_translation = stored['translation']
            correct_tense = stored['tense']
            base_verb = stored.get('base_verb', conjugated_form)

            student_answer = request.translation.strip()

            if history.direction == 'reverse':
                # Script validation for reverse mode
                if script != 'latin' and not validate_script(student_answer, script):
                    script_name = 'Cyrillic' if script == 'cyrillic' else script
                    raise HTTPException(status_code=400, detail=f"Please write your answer in {script_name} script.")
                # Reverse mode (EN→target): user sees English, types target language conjugated form
                correct_display = conjugated_form
                sa_lower = student_answer.lower()
                cf_lower = conjugated_form.lower()
                translation_correct = sa_lower == cf_lower
                if not translation_correct:
                    translation_correct = accent_lenient_match(student_answer, conjugated_form, target_is_spanish=True, accent_words=accent_set)
            else:
                # Forward mode (ES→EN): user sees Spanish, types English translation
                correct_display = english_translation
                verb_check = ai_provider.validate_verb_translation(
                    conjugated_form, base_verb, english_translation, student_answer,
                    language_info=lang_info
                )
                translation_correct = verb_check.get('correct', False)

            # Check tense
            selected_tense = (request.selected_tense or '').lower()
            tense_correct = selected_tense == correct_tense

            # Score translation and tense independently
            # Translation is the primary score (worth 80 points), tense is secondary (worth 20 points)
            translation_score = 80 if translation_correct else 0
            tense_score = 20 if tense_correct else 0
            score = translation_score + tense_score

            if translation_correct and tense_correct:
                evaluation = 'Correct!'
            elif translation_correct:
                evaluation = f'Translation correct! But the tense is {correct_tense}, not {selected_tense}.'
            elif tense_correct:
                evaluation = f'Tense correct, but the translation should be: {correct_display}'
            else:
                evaluation = f'The translation is "{correct_display}" and the tense is {correct_tense}.'

            judgement = {
                'score': score,
                'correct_translation': correct_display,
                'correct_tense': correct_tense,
                'evaluation': evaluation,
                'translation_correct': translation_correct,
                'tense_correct': tense_correct,
                'vocabulary_breakdown': [[conjugated_form, english_translation, 'verb', translation_correct]]
            }
            judge_ms = 0

            # Log sentence mismatch as warning (don't block submission — evaluation uses server state)
            expected_sentence = english_translation if history.direction == 'reverse' else conjugated_form
            if expected_sentence != request.sentence:
                logger.warning(f"Verb sentence mismatch: expected={expected_sentence!r}, got={request.sentence!r}, direction={history.direction}")

        elif is_multi_vocab:
            # Multi-word vocabulary challenge (4 words, stored as english keys)
            is_reverse = current_round.sentence.startswith("VOCAB4R:")
            prefix = "VOCAB4R:" if is_reverse else "VOCAB4:"
            rest = current_round.sentence[len(prefix):]
            parts = rest.split(":", 1)
            if len(parts) != 2:
                raise HTTPException(status_code=400, detail="Invalid multi-vocab challenge format")

            category = parts[0]
            english_keys = parts[1].split(",")
            logger.info(f"Multi-vocab challenge: category={category}, english_keys={english_keys}, reverse={is_reverse}")

            # Look up each english key to get word/alternatives
            vocab_items = []
            for eng_key in english_keys:
                item = storage.get_vocab_item_by_english(category, eng_key, language=lang_code)
                if item:
                    vocab_items.append(item)
                else:
                    vocab_items.append({'english': eng_key, 'word': eng_key, 'alternatives': eng_key})

            # Score each word (25 points each)
            word_results = []
            total_score = 0
            translations = request.translations or []

            for i, item in enumerate(vocab_items):
                student_answer = translations[i].strip() if i < len(translations) else ''
                spanish_word = item['word']
                alternatives = item['alternatives']

                if is_reverse:
                    # Reverse: English shown, user types Spanish word
                    is_correct = student_answer.lower() == spanish_word.lower()
                    if not is_correct:
                        is_correct = accent_lenient_match(student_answer, spanish_word, target_is_spanish=True, accent_words=accent_set)
                    word_results.append({
                        'word': alternatives,  # English shown
                        'correct_answer': spanish_word,  # Spanish expected
                        'student_answer': student_answer,
                        'is_correct': is_correct
                    })
                else:
                    # Forward: Spanish shown, user types English translation
                    correct_answers = [t.strip().lower() for t in alternatives.split(',')]
                    is_correct = student_answer.lower() in correct_answers
                    if not is_correct:
                        is_correct = any(accent_lenient_match(student_answer, a, target_is_spanish=False, accent_words=accent_set) for a in correct_answers)
                    word_results.append({
                        'word': spanish_word,  # Spanish shown
                        'correct_answer': alternatives,
                        'student_answer': student_answer,
                        'is_correct': is_correct
                    })

                if is_correct:
                    total_score += 25

                # Record per-word vocab progress by english key
                history.record_vocab_result(category, item['english'], is_correct)

            all_correct = total_score == 100
            correct_str = ', '.join(f"{w['word']}={w['correct_answer']}" for w in word_results if not w['is_correct'])
            if all_correct:
                evaluation = 'All correct!'
            else:
                evaluation = f'Incorrect: {correct_str}'

            if is_reverse:
                # Reverse mode: English shown, Spanish expected
                vocab_breakdown = [[item['alternatives'], item['word'], category, word_results[i]['is_correct']] for i, item in enumerate(vocab_items)]
                correct_trans = ', '.join(item['word'] for item in vocab_items)
            else:
                # Forward mode: Spanish shown, English expected
                vocab_breakdown = [[item['word'], item['alternatives'], category, word_results[i]['is_correct']] for i, item in enumerate(vocab_items)]
                correct_trans = ', '.join(item['alternatives'] for item in vocab_items)

            judgement = {
                'score': total_score,
                'correct_translation': correct_trans,
                'evaluation': evaluation,
                'vocabulary_breakdown': vocab_breakdown
            }
            judge_ms = 0

        elif is_vocab_challenge:
            # Single-word vocabulary category challenge: simple matching
            # Format: VOCAB:category:english_key or VOCABR:category:english_key
            is_reverse = current_round.sentence.startswith("VOCABR:")
            parts = current_round.sentence.split(":", 2)
            if len(parts) != 3:
                raise HTTPException(status_code=400, detail="Invalid vocab challenge format")

            category = parts[1]
            english_key = parts[2]
            logger.info(f"Vocab challenge: category={category}, english_key={english_key}, reverse={is_reverse}")

            # Look up the vocabulary item by english key
            item = storage.get_vocab_item_by_english(category, english_key)
            if item:
                spanish_word = item['word']
                alternatives = item['alternatives']
            else:
                spanish_word = english_key
                alternatives = english_key

            if is_reverse:
                # en->es: user sees English, types Spanish word
                correct_answers = [spanish_word.strip().lower()]
                correct_display = spanish_word
                displayed_word = english_key
            else:
                # es->en: user sees Spanish, types English translation
                correct_answers = [t.strip().lower() for t in alternatives.split(',')]
                correct_display = alternatives
                displayed_word = spanish_word

            # Check if translation matches (case-insensitive)
            student_answer = request.translation.strip().lower()
            is_correct = student_answer in correct_answers
            if not is_correct:
                target_is_spanish = is_reverse
                is_correct = any(accent_lenient_match(student_answer, a, target_is_spanish=target_is_spanish, accent_words=accent_set) for a in correct_answers)

            score = 100 if is_correct else 0
            judgement = {
                'score': score,
                'correct_translation': correct_display,
                'evaluation': 'Correct!' if is_correct else f'The correct translation is: {correct_display}',
                'vocabulary_breakdown': [[spanish_word, alternatives, category, is_correct]]
            }
            judge_ms = 0

            # Record vocab challenge result by english key
            history.record_vocab_result(category, english_key, is_correct)

            # Log sentence mismatch as warning (don't block submission — evaluation uses server state)
            if displayed_word != request.sentence:
                logger.warning(f"Vocab sentence mismatch: expected={displayed_word!r}, got={request.sentence!r}")

        elif is_word_challenge:
            # Word challenge: get translation from storage or AI
            word = current_round.sentence[5:]  # Remove "WORD:" prefix
            logger.info(f"Word challenge for word: {word}")

            if history.direction == 'reverse':
                # Reverse mode (EN→target): user saw English translation, correct answer is the target word itself
                word_info = history.words.get(word, {})
                correct_translation = word  # The target language word IS the answer
                word_type = word_info.get('type') or 'unknown'
                logger.info(f"Reverse word challenge: {word_info.get('translation', '?')} -> {correct_translation} ({word_type})")
            else:
                # Normal mode (ES→EN): word is Spanish, correct answer is English
                # First check persistent storage for translation
                stored_translation = storage.get_word_translation(word, language=lang_code)

                if stored_translation:
                    logger.info(f"Using stored translation: {stored_translation}")
                    correct_translation = stored_translation['translation']
                    word_type = stored_translation['type']
                else:
                    # Query AI for translation and store it
                    logger.info(f"Querying AI for translation of: {word}")
                    ai_result = ai_provider.get_word_translation(word, language_info=lang_info)
                    if ai_result:
                        correct_translation = ai_result['translation']
                        word_type = ai_result['type']
                        # Save to persistent storage
                        storage.save_word_translation(word, correct_translation, word_type, language=lang_code)
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

            # Check if translation matches (case-insensitive, with plural tolerance)
            student_answer = request.translation.strip().lower()

            def normalize_word(w: str) -> str:
                """Normalize word for comparison (handle common plural/singular forms)."""
                w = w.strip()
                if w.endswith('ies'):
                    return w[:-3] + 'y'  # flies -> fly
                if w.endswith('es'):
                    return w[:-2]  # watches -> watch
                if w.endswith('s') and len(w) > 2:
                    return w[:-1]  # marks -> mark
                return w

            # Check exact match first, then normalized match, then accent-lenient
            target_is_spanish = history.direction == 'reverse'
            is_correct = student_answer in correct_answers
            if not is_correct:
                # Try normalized comparison (singular/plural tolerance)
                normalized_student = normalize_word(student_answer)
                normalized_correct = {normalize_word(a) for a in correct_answers}
                is_correct = normalized_student in normalized_correct or student_answer in normalized_correct
            if not is_correct:
                # Try accent-lenient matching
                is_correct = any(accent_lenient_match(student_answer, a, target_is_spanish=target_is_spanish, accent_words=accent_set) for a in correct_answers)

            score = 100 if is_correct else 0
            judgement = {
                'score': score,
                'correct_translation': correct_translation,
                'evaluation': 'Correct!' if is_correct else f'The correct translation is: {correct_translation}',
                'vocabulary_breakdown': [[word, correct_translation, word_type, is_correct]]
            }
            judge_ms = 0

            # Log sentence mismatch as warning (don't block submission — evaluation uses server state)
            if word != request.sentence:
                logger.warning(f"Word sentence mismatch: expected={word!r}, got={request.sentence!r}")
        else:
            # Regular sentence: AI validation uses current_round.sentence (server state)
            if current_round.sentence != request.sentence:
                logger.warning(f"Regular sentence mismatch: expected={current_round.sentence!r}, got={request.sentence!r}")

            # Script validation for reverse mode (student types target language)
            if history.direction == 'reverse' and script != 'latin':
                if not validate_script(request.translation, script):
                    script_name = 'Cyrillic' if script == 'cyrillic' else script
                    raise HTTPException(
                        status_code=400,
                        detail=f"Please write your answer in {script_name} script."
                    )

            judgement, judge_ms = ai_provider.validate_translation(
                current_round.sentence,
                request.translation,
                story_context=history.current_story,
                direction=history.direction,
                language_info=lang_info
            )

        current_round.translation = request.translation

        # Challenges have separate scoring, don't affect level progress
        if is_verb_challenge or is_vocab_challenge or is_word_challenge or is_synonym_challenge:
            current_round.judgement = judgement
            current_round.judge_ms = judge_ms
            current_round.evaluated = True
            history.total_completed += 1

            # Record challenge stats
            challenge_type = 'verb' if is_verb_challenge else ('synonym' if is_synonym_challenge else ('vocab' if is_vocab_challenge else 'word'))
            if is_verb_challenge:
                # For verb challenges, count translation correctness for the word stat
                is_fully_correct = judgement.get('translation_correct', False)
            else:
                is_fully_correct = judgement.get('score', 0) >= 100
            history.record_challenge_result(challenge_type, is_fully_correct)

            # For word challenges, also update word tracking for learning
            if is_word_challenge:
                history.update_words(current_round, request.hint_words or [])
                # Mark word as challenge_passed so it won't reappear
                # (unless missed later in sentence translation)
                if is_fully_correct:
                    word = current_round.sentence[5:]  # Remove "WORD:" prefix
                    if word in history.words:
                        history.words[word]['challenge_passed'] = True

            # Store the evaluated round so it shows on next page
            history.last_evaluated_round = current_round
            history.last_level_changed = False

            save_history(request.user_id)

            # Log translation submission and result for challenges
            ms = int((time.time() - start_time) * 1000)
            log_event('translation.submit', request.user_id,
                      sentence=request.sentence,
                      translation=request.translation,
                      challenge_type=challenge_type,
                      hint_used=request.hint_used,
                      ms=ms)
            log_event('translation.result', request.user_id,
                      score=current_round.get_score(),
                      challenge_type=challenge_type,
                      correct_translation=judgement.get('correct_translation', ''),
                      ms=judge_ms)

            response = TranslationResponse(
                score=current_round.get_score(),
                correct_translation=judgement.get('correct_translation', ''),
                evaluation=judgement.get('evaluation', ''),
                vocabulary_breakdown=judgement.get('vocabulary_breakdown', []),
                judge_ms=judge_ms,
                level_changed=False,
                new_level=history.difficulty,
                change_type=None
            )

            # Include per-word results for multi-word vocab challenges
            if is_multi_vocab:
                response.word_results = word_results

            return response

        # Regular sentences affect level progress
        level_info = history.process_evaluation(judgement, judge_ms, current_round, request.hint_words, request.hint_used)
        save_history(request.user_id)

        # Log translation submission and result
        ms = int((time.time() - start_time) * 1000)
        log_event('translation.submit', request.user_id,
                  sentence=request.sentence,
                  translation=request.translation,
                  hint_used=request.hint_used,
                  hint_words=request.hint_words,
                  ms=ms)
        validate_ai = ai_provider.get_last_call_info()
        log_event('translation.result', request.user_id,
                  score=current_round.get_score(),
                  correct_translation=judgement.get('correct_translation', ''),
                  ms=judge_ms,
                  ai_used=True,
                  model_name=validate_ai.get('model_name'),
                  model_tokens=validate_ai.get('model_tokens'),
                  model_ms=validate_ai.get('model_ms'))

        # Log level change if it occurred
        if level_info['level_changed']:
            log_event('level.change', request.user_id,
                      old_level=level_info['new_level'] - 1 if level_info['change_type'] == 'up' else level_info['new_level'] + 1,
                      new_level=level_info['new_level'],
                      direction=level_info['change_type'])

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
        logger.error(traceback.format_exc())
        log_event('error.server', request.user_id,
                  endpoint='/api/translate',
                  error_type=type(e).__name__,
                  error_message=str(e),
                  traceback=traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal error: {type(e).__name__}: {str(e)}")


@app.post("/api/hint", response_model=HintResponse)
async def get_hint(request: HintRequest):
    """Get a hint for the current sentence."""
    try:
        return await _get_hint_inner(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_hint: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        log_event('error.server', request.user_id,
                  endpoint='/api/hint',
                  error_type=type(e).__name__,
                  error_message=str(e),
                  traceback=traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


async def _get_hint_inner(request: HintRequest):
    start_time = time.time()
    history = get_history(request.user_id)

    lang_info = get_user_language_info(request.user_id)
    hint = ai_provider.get_hint(request.sentence, history.correct_words, direction=history.direction, partial_translation=request.partial_translation, language_info=lang_info)

    # Sanitize hint entries: discard arrays with null/None/"null" values
    def valid_entry(entry):
        return (isinstance(entry, list) and len(entry) >= 2
                and entry[0] is not None and entry[0] != 'null'
                and entry[1] is not None and entry[1] != 'null')

    if hint:
        for key in ('noun', 'verb', 'adjective'):
            if key in hint and not valid_entry(hint.get(key)):
                hint[key] = None

    # Log hint request
    words_revealed = []
    if hint:
        for key in ('noun', 'verb', 'adjective'):
            if hint.get(key):
                words_revealed.append(hint[key][0])
    hint_ai = ai_provider.get_last_call_info()
    ms = int((time.time() - start_time) * 1000)
    log_event('hint.request', request.user_id,
              sentence=request.sentence,
              words_revealed=words_revealed,
              ms=ms,
              ai_used=True,
              model_name=hint_ai.get('model_name'),
              model_tokens=hint_ai.get('model_tokens'),
              model_ms=hint_ai.get('model_ms'))

    if not hint:
        return HintResponse(noun=None, verb=None, adjective=None)

    return HintResponse(
        noun=hint.get('noun'),
        verb=hint.get('verb'),
        adjective=hint.get('adjective')
    )


@app.post("/api/verb-hint", response_model=VerbHintResponse)
async def get_verb_hint(request: VerbHintRequest):
    """Get conjugation rules hint for a verb challenge."""
    try:
        return await _get_verb_hint_inner(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_verb_hint: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        log_event('error.server', request.user_id,
                  endpoint='/api/verb-hint',
                  error_type=type(e).__name__,
                  error_message=str(e),
                  traceback=traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


async def _get_verb_hint_inner(request: VerbHintRequest):
    start_time = time.time()
    history = get_history(request.user_id)
    lang_code = history.language
    lang_info = get_user_language_info(request.user_id)

    # Extract conjugated form from "VERB:conjugated_form"
    sentence = request.sentence
    if sentence.startswith("VERB:"):
        conjugated_form = sentence[5:]
    else:
        conjugated_form = sentence

    # Look up the verb info to get the tense
    stored = storage.get_verb_conjugation(conjugated_form, language=lang_code)
    if not stored:
        raise HTTPException(status_code=404, detail="Verb conjugation not found")

    tense = stored['tense']
    base_verb = stored.get('base_verb', conjugated_form)

    # Check DB cache for conjugation rules
    rules = storage.get_verb_conjugation_rules(lang_code, tense)

    if not rules:
        # Generate via Gemini and cache
        rules = ai_provider.generate_conjugation_rules(tense, language_info=lang_info)
        if rules:
            storage.save_verb_conjugation_rules(lang_code, tense, rules)

    if not rules:
        raise HTTPException(status_code=500, detail="Could not generate conjugation rules")

    ms = int((time.time() - start_time) * 1000)
    hint_ai = ai_provider.get_last_call_info()
    log_event('verb_hint.request', request.user_id,
              conjugated_form=conjugated_form,
              tense=tense,
              base_verb=base_verb,
              cached=storage.get_verb_conjugation_rules(lang_code, tense) is not None,
              ms=ms,
              ai_used=bool(hint_ai.get('model_name')),
              model_name=hint_ai.get('model_name'),
              model_tokens=hint_ai.get('model_tokens'),
              model_ms=hint_ai.get('model_ms'))

    return VerbHintResponse(rules=rules, tense=tense, base_verb=base_verb)


@app.post("/api/downgrade")
async def downgrade_level(user_id: str = "default"):
    """Voluntarily go back to the previous difficulty level, resetting score progress."""
    history = get_history(user_id)
    old_level = history.difficulty

    if old_level <= MIN_DIFFICULTY:
        return {"success": False, "error": "Already at the lowest level"}

    history.demote_level()
    save_history(user_id)

    log_event('level.downgrade', user_id,
              old_level=old_level,
              new_level=history.difficulty)

    return {"success": True, "new_level": history.difficulty}


@app.post("/api/switch-direction")
async def switch_direction(user_id: str = "default"):
    """Switch between normal (ES→EN) and reverse (EN→ES) translation direction."""
    history = get_history(user_id)
    old_direction = history.direction

    history.switch_direction()
    save_history(user_id)

    # Clear pending stories (wrong direction)
    pending_stories.pop(user_id, None)

    # Trigger background story generation for new direction
    trigger_background_story(user_id, history)

    log_event('direction.switch', user_id,
              old_direction=old_direction,
              new_direction=history.direction)

    return {"success": True, "direction": history.direction}


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
    """Get Gemini API usage statistics from both providers."""
    # Get stats from both providers
    flash_stats = ai_provider.get_stats()
    pro_stats = story_provider.get_stats()

    # Merge stats - flash has validate/hint/word_translation/verb_analysis, pro has story
    merged = {}
    for key, value in flash_stats.items():
        if key != 'total':
            merged[key] = value
    for key, value in pro_stats.items():
        if key != 'total':
            merged[key] = value

    # Recalculate totals
    totals = {'calls': 0, 'total_ms': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    for key, stats in merged.items():
        for tkey in totals:
            totals[tkey] += stats.get(tkey, 0)

    calls = totals['calls']
    merged['total'] = {
        **totals,
        'avg_ms': round(totals['total_ms'] / calls, 1) if calls > 0 else 0,
        'avg_tokens': round(totals['total_tokens'] / calls, 1) if calls > 0 else 0
    }

    return merged


@app.get("/api/events/apps")
async def get_event_apps():
    """Get all unique app names from events."""
    if not hasattr(storage, 'get_app_names'):
        return {"apps": []}

    return {"apps": storage.get_app_names()}


@app.get("/api/events/users")
async def get_event_users():
    """Get all unique user IDs from events."""
    if not hasattr(storage, 'get_event_users'):
        return {"users": []}

    return {"users": storage.get_event_users()}


@app.get("/api/events/stats")
async def get_event_stats(user_id: str = None):
    """Get aggregated event statistics."""
    if not hasattr(storage, 'get_user_stats'):
        return {"error": "Event logging not available with current storage"}

    if user_id:
        return storage.get_user_stats(user_id)
    else:
        return storage.get_global_stats()


@app.get("/api/events/recent")
async def get_recent_events(user_id: str = None, event_type: str = None,
                            app_name: str = None, limit: int = 100):
    """Get recent events with optional filters."""
    if not hasattr(storage, 'get_events'):
        return {"error": "Event logging not available with current storage"}

    events = storage.get_events(user_id, event_type, app_name, limit)
    # Convert datetime objects to strings for JSON serialization
    for event in events:
        if 'timestamp' in event and hasattr(event['timestamp'], 'isoformat'):
            event['timestamp'] = event['timestamp'].isoformat()
    return {"events": events}


@app.get("/api/perf")
async def get_perf_stats(app_name: str = None):
    """Get performance statistics grouped by event type."""
    if not hasattr(storage, 'get_perf_stats'):
        return {"error": "Performance stats not available with current storage"}

    stats = storage.get_perf_stats(app_name)
    # Convert Decimal types to float for JSON serialization
    for stat in stats:
        for key, value in stat.items():
            if hasattr(value, '__float__'):
                stat[key] = float(value) if value is not None else None
    return {"stats": stats}


@app.post("/api/error")
async def log_client_error(request: ErrorLogRequest):
    """Log a client-side error to the database."""
    logger.error(f"Client error: {request.endpoint} - {request.error} (user={request.user_id}, status={request.status})")
    log_event('error.client', request.user_id or 'unknown',
              endpoint=request.endpoint,
              error_message=request.error,
              status_code=request.status,
              context=request.context)
    return {"logged": True}


@app.get("/api/errors/recent")
async def get_recent_errors(user_id: str = None, limit: int = 20):
    """Get recent errors (both client and server)."""
    if not hasattr(storage, 'get_events'):
        return {"errors": []}

    events = storage.get_events(user_id, event_type=None, app_name='tongue', limit=limit * 5)
    errors = []
    for event in events:
        if event.get('event', '').startswith('error.'):
            if 'timestamp' in event and hasattr(event['timestamp'], 'isoformat'):
                event['timestamp'] = event['timestamp'].isoformat()
            errors.append(event)
            if len(errors) >= limit:
                break
    return {"errors": errors}


def create_app():
    """Factory function for creating the app (useful for testing)."""
    return app

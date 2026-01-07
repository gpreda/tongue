"""FastAPI server for tongue application."""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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


class HintRequest(BaseModel):
    sentence: str
    user_id: str = "default"


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
    level_scores: list[int]
    good_score_count: int
    poor_score_count: int
    story_sentences_remaining: int
    progress_display: str


class NextSentenceResponse(BaseModel):
    sentence: str
    difficulty: int
    story: str
    sentences_remaining: int
    progress_display: str
    has_previous_evaluation: bool
    previous_evaluation: Optional[dict]


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

    config = storage.load_config()

    api_key = config.get('gemini_api_key')
    if not api_key:
        raise RuntimeError("gemini_api_key not found in config file")

    ai_provider = GeminiProvider(api_key)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "tongue-api", "language": LANGUAGE}


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
        progress_display=history.get_progress_display()
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
    """Get the next sentence for translation."""
    history = get_history(user_id)

    # Generate story if needed
    if history.needs_new_story():
        story, ms = ai_provider.generate_story(
            history.correct_words,
            history.difficulty
        )
        history.set_story(story, history.difficulty, ms)
        save_history(user_id)

    # Get next sentence
    round = history.get_next_sentence()
    if not round:
        raise HTTPException(status_code=500, detail="No sentences available")

    save_history(user_id)

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

    return NextSentenceResponse(
        sentence=round.sentence,
        difficulty=round.difficulty,
        story=history.current_story or "",
        sentences_remaining=len(history.story_sentences),
        progress_display=history.get_progress_display(),
        has_previous_evaluation=has_previous,
        previous_evaluation=previous_eval
    )


@app.post("/api/translate", response_model=TranslationResponse)
async def submit_translation(request: TranslationRequest):
    """Submit a translation for evaluation."""
    history = get_history(request.user_id)

    # Find the current round (last in rounds list)
    if not history.rounds:
        raise HTTPException(status_code=400, detail="No active round")

    current_round = history.rounds[-1]
    if current_round.evaluated:
        raise HTTPException(status_code=400, detail="Round already evaluated")

    if current_round.sentence != request.sentence:
        raise HTTPException(status_code=400, detail="Sentence mismatch")

    # Evaluate translation
    judgement, judge_ms = ai_provider.validate_translation(
        request.sentence,
        request.translation
    )

    current_round.translation = request.translation
    level_info = history.process_evaluation(judgement, judge_ms, current_round)
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


@app.get("/api/missed-words")
async def get_missed_words(user_id: str = "default", limit: int = 20):
    """Get words that need more practice."""
    history = get_history(user_id)

    sorted_missed = sorted(
        history.missed_words.items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )[:limit]

    return {
        "total": len(history.missed_words),
        "words": [
            {"word": word, "english": info["english"], "count": info["count"]}
            for word, info in sorted_missed
        ]
    }


@app.get("/api/mastered-words")
async def get_mastered_words(user_id: str = "default", limit: int = 50):
    """Get mastered words."""
    history = get_history(user_id)

    return {
        "total": len(history.correct_words),
        "words": history.correct_words[-limit:]
    }


def create_app():
    """Factory function for creating the app (useful for testing)."""
    return app

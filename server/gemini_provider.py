"""Gemini AI provider implementation."""

import ast
import logging
import random
import time
import google.generativeai as genai

from core.interfaces import AIProvider
from core.config import LANGUAGE, MAX_DIFFICULTY, STORY_SENTENCE_COUNT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SEASONS = ["spring", "summer", "autumn", "winter"]


class GeminiProvider(AIProvider):
    """Gemini AI provider implementation."""

    def __init__(self, api_key: str, model_name: str = 'gemini-2.0-flash'):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.chat = self.model.start_chat(history=[])
        self.model_name = model_name

    def _execute_chat(self, prompt: str) -> tuple[str, int]:
        start_time = time.time()
        response = self.chat.send_message(prompt)
        end_time = time.time()
        ms = int((end_time - start_time) * 1000)
        return (response.text, ms)

    def _sanitize_judgement(self, judgement: str) -> str:
        s = judgement[judgement.find('{'):judgement.rfind('}')+1]
        s = s.replace('false', 'False')
        s = s.replace('true', 'True')
        return s

    def generate_story(self, correct_words: list, difficulty: int) -> tuple[str, int]:
        avoided_words = correct_words[-100:] if len(correct_words) > 100 else correct_words

        # Generate random elements for story diversity
        seed = int(time.time() * 1000) % 100000
        number = random.randint(0, 10)
        day = random.choice(DAYS_OF_WEEK)
        season = random.choice(SEASONS)
        letter = random.choice("ABCDEFGHJLMNPRST")

        prompt = f"""
            Story ID: {seed}

            Write a short, engaging story in {LANGUAGE} with approximately {STORY_SENTENCE_COUNT} sentences.

            MANDATORY elements (you MUST include ALL of these):
            - The number {number} appears meaningfully in the story
            - The story takes place on a {day} in {season}
            - The first sentence must include a noun that starts with the letter "{letter}"

            Requirements:
            - Difficulty level: {difficulty} out of {MAX_DIFFICULTY}
              Level 1 = beginner (simple vocabulary, present tense, basic grammar)
              Level 5 = intermediate (varied vocabulary, multiple tenses, compound sentences)
              Level 10 = expert (advanced vocabulary, subjunctive, idioms, complex grammar)
            - Each sentence should be 5-20 words long
            - Use vocabulary and grammar appropriate for level {difficulty}
            - Try to avoid using these nouns (the student already knows them): {', '.join(avoided_words) if avoided_words else 'none'}

            Write only the story text, no titles, no translations, no explanations.
            Each sentence should end with proper punctuation (. ! ?).
        """
        return self._execute_chat(prompt)

    def validate_translation(self, sentence: str, translation: str) -> tuple[dict, int]:
        prompt = f"""
            You are evaluating a student's English translation of a {LANGUAGE} sentence.

            {LANGUAGE} Sentence: "{sentence}"
            Student's Translation: "{translation}"

            STEP 1 - WORD-BY-WORD MAPPING (do this carefully):
            For EACH content word in the {LANGUAGE} sentence:
            a) List the correct English translation(s)
            b) Find what English word the student actually used for it
            c) Determine if they match

            Example analysis:
            - "escuela" → correct: "school" → student used: "stairs" → MISMATCH (mistranslated)
            - "caminan" → correct: "walk/walking" → student used: "walk" → MATCH

            STEP 2 - EVALUATION RULES:
            1. A word is CORRECTLY translated if the student used a valid English equivalent.
               Multiple English words can be valid translations (e.g., "vista" can be "view", "sight", "vision").
            2. A word is INCORRECTLY translated if:
               - Left in {LANGUAGE} (not translated at all, e.g., "casa" instead of "house")
               - Translated to a wrong meaning
               - Omitted entirely
            3. Do NOT penalize the student for:
               - Using synonyms (these are valid translations)
               - Minor word order differences that don't change meaning
               - Using articles (a/an/the) flexibly
               - Capitalization differences (e.g., "monday" instead of "Monday", or not capitalizing the first word)
               - Missing or different punctuation (periods, commas, exclamation marks)
            4. Only proper nouns (names of people, places, brands) may remain in original form.
            5. If the translation is semantically correct with all words properly translated, the score should be 100.

            STEP 3 - RESPOND with a Python dictionary:

            'vocabulary_breakdown': A list of lists, one for each meaningful word/phrase:
              [0] The {LANGUAGE} word/phrase from the original sentence
              [1] The correct English translation(s) for this word in context
              [2] Part of speech (noun, verb, adjective, etc.)
              [3] Boolean: True ONLY if the student's translation contains a valid English equivalent.
                  False if: the word was mistranslated (e.g., "escuela"→"stairs" instead of "school"),
                  left untranslated, or omitted entirely.
                  IMPORTANT: Check the actual English word the student used, not just sentence meaning.

            'correct_translation': A proper English translation of the sentence.

            'score': Integer 1-100. Base it on:
              - What percentage of words were correctly translated
              - Deduct heavily for untranslated {LANGUAGE} words left as-is
              - Deduct for wrong meanings
              - Minor deductions for grammar issues

            'evaluation': Brief explanation. Be ACCURATE - do not claim a word was "added" if it's
            a valid translation of a word in the original sentence. Only mention actual errors.

            Return ONLY the dictionary, no other text, no markdown formatting.
        """
        response, ms = self._execute_chat(prompt)
        sanitized = self._sanitize_judgement(response)

        try:
            judgement = ast.literal_eval(sanitized)

            # Validate required keys are present
            required_keys = ['score', 'correct_translation', 'evaluation', 'vocabulary_breakdown']
            missing_keys = [k for k in required_keys if k not in judgement]
            if missing_keys:
                logger.warning(f"AI response missing keys: {missing_keys}")
                logger.warning(f"Raw response:\n{response}")
                logger.warning(f"Sanitized response:\n{sanitized}")
                # Fill in missing keys with defaults
                if 'score' not in judgement:
                    judgement['score'] = 50
                if 'correct_translation' not in judgement:
                    judgement['correct_translation'] = 'Translation unavailable'
                if 'evaluation' not in judgement:
                    judgement['evaluation'] = 'Evaluation unavailable'
                if 'vocabulary_breakdown' not in judgement:
                    judgement['vocabulary_breakdown'] = []

            # Validate score is a number
            if not isinstance(judgement.get('score'), (int, float)):
                logger.warning(f"Invalid score type: {type(judgement.get('score'))} = {judgement.get('score')}")
                judgement['score'] = 50

        except (SyntaxError, ValueError) as e:
            logger.error(f"Failed to parse judgement: {e}")
            logger.error(f"Raw response:\n{response}")
            logger.error(f"Sanitized response:\n{sanitized}")

            # Try to diagnose the issue
            if '{' not in response:
                logger.error("Diagnosis: No opening brace '{' found in response")
            elif '}' not in response:
                logger.error("Diagnosis: No closing brace '}' found in response")
            elif sanitized.count('{') != sanitized.count('}'):
                logger.error(f"Diagnosis: Mismatched braces - {{ count: {sanitized.count('{')}, }} count: {sanitized.count('}')}")
            elif "'score'" not in response and '"score"' not in response:
                logger.error("Diagnosis: 'score' key not found in response")
            else:
                logger.error("Diagnosis: Unknown parsing issue - possibly malformed Python dict syntax")

            # Return a fallback response
            judgement = {
                'score': 50,
                'correct_translation': 'Unable to parse response',
                'evaluation': 'Error parsing AI response. Please try again.',
                'vocabulary_breakdown': []
            }
        return (judgement, ms)

    def get_hint(self, sentence: str, correct_words: list) -> dict | None:
        prompt = f"""
            Analyze this {LANGUAGE} sentence and provide a hint for translation:

            Sentence: "{sentence}"

            Already known words (do NOT include these): {', '.join(correct_words[-50:]) if correct_words else 'none'}

            Find the most challenging/uncommon noun, verb, and adjective from the sentence.
            Prioritize less common vocabulary that a student would most likely need help with.
            AVOID basic/common words like: quiere, tiene, es, está, hay, va, hace, puede, debe, dice, sabe, viene, ser, estar, tener, ir, hacer, poder, deber, decir, saber, venir, el, la, un, una, los, las.
            Provide their English translations.

            Respond with ONLY a Python dictionary in this exact format:
            {{
                'noun': ['spanish_noun', 'english_translation'],
                'verb': ['spanish_verb', 'english_translation'],
                'adjective': ['spanish_adjective', 'english_translation']
            }}

            If any part of speech is not available (all are known or not present), use null for that entry:
            {{'noun': null, 'verb': ['spanish_verb', 'english_translation'], 'adjective': null}}

            Return ONLY the dictionary, no other text.
        """
        response, ms = self._execute_chat(prompt)
        raw_response = response
        try:
            response = response.strip()
            response = response.replace('null', 'None')
            response = response.replace('```python', '').replace('```', '')
            hint = eval(response)

            # Validate structure
            if not isinstance(hint, dict):
                logger.warning(f"Hint response is not a dict: {type(hint)}")
                logger.warning(f"Raw response:\n{raw_response}")
                return None

            return hint
        except Exception as e:
            logger.error(f"Failed to parse hint: {e}")
            logger.error(f"Raw response:\n{raw_response}")
            logger.error(f"Cleaned response:\n{response}")

            # Diagnose
            if '{' not in raw_response:
                logger.error("Diagnosis: No opening brace '{' found")
            elif '}' not in raw_response:
                logger.error("Diagnosis: No closing brace '}' found")

            return None

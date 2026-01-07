"""Gemini AI provider implementation."""

import time
import google.generativeai as genai

from core.interfaces import AIProvider
from core.config import LANGUAGE, MAX_DIFFICULTY, STORY_SENTENCE_COUNT


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

        prompt = f"""
            Write a short, engaging story in {LANGUAGE} with approximately {STORY_SENTENCE_COUNT} sentences.
            The story should be a random, creative narrative (adventure, mystery, daily life, fantasy, etc.).

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

            STEP 1 - ANALYZE THE ORIGINAL SENTENCE FIRST:
            Before evaluating, carefully identify ALL words and their meanings in the {LANGUAGE} sentence.
            Consider all possible valid English translations for each word, including synonyms.

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
            4. Only proper nouns (names of people, places, brands) may remain in original form.

            STEP 3 - RESPOND with a Python dictionary:

            'vocabulary_breakdown': FIRST, create this list analyzing the ORIGINAL {LANGUAGE} sentence.
            A list of lists, one for each meaningful word/phrase:
              [0] The {LANGUAGE} word/phrase from the original sentence
              [1] The correct English translation(s) for this word in context
              [2] Part of speech (noun, verb, adjective, etc.)
              [3] Boolean: True if the student correctly translated this word (including valid synonyms).
                  False only if: untranslated, mistranslated, or omitted.

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
        judgement = self._sanitize_judgement(response)
        judgement = eval(judgement)
        return (judgement, ms)

    def get_hint(self, sentence: str, correct_words: list) -> dict | None:
        prompt = f"""
            Analyze this {LANGUAGE} sentence and provide a hint for translation:

            Sentence: "{sentence}"

            Already known words (do NOT include these): {', '.join(correct_words[-50:]) if correct_words else 'none'}

            Find ONE noun, ONE verb, and ONE adjective from the sentence that are NOT in the known words list.
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
        try:
            response = response.strip()
            response = response.replace('null', 'None')
            response = response.replace('```python', '').replace('```', '')
            return eval(response)
        except:
            return None

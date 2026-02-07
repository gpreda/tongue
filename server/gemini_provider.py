"""Gemini AI provider implementation."""

import ast
import logging
import random
import time
import google.generativeai as genai

from core.interfaces import AIProvider
from core.config import MAX_DIFFICULTY, STORY_SENTENCE_COUNT

# Default language info (Spanish) used when no language_info is provided
_DEFAULT_LANGUAGE_INFO = {
    'code': 'es',
    'name': 'Español',
    'english_name': 'Spanish',
    'script': 'latin',
    'tenses': ['present', 'preterite', 'imperfect', 'future', 'conditional', 'subjunctive'],
    'accent_words': []
}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SEASONS = ["spring", "summer", "autumn", "winter"]


class GeminiProvider(AIProvider):
    """Gemini AI provider implementation."""

    def __init__(self, api_key: str, model_name: str = 'gemini-2.0-flash', storage=None):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.chat = self.model.start_chat(history=[])
        self.model_name = model_name
        self.storage = storage
        self.provider_name = f"gemini_{model_name.replace('-', '_').replace('.', '_')}"

        # Default stats structure
        default_stats = {
            'story': {'calls': 0, 'total_ms': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
            'validate': {'calls': 0, 'total_ms': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
            'hint': {'calls': 0, 'total_ms': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
            'word_translation': {'calls': 0, 'total_ms': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
            'verb_analysis': {'calls': 0, 'total_ms': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        }

        # Load stats from storage or use defaults
        if storage:
            loaded_stats = storage.load_api_stats(self.provider_name)
            if loaded_stats:
                # Merge loaded stats with defaults to handle new call types
                self.stats = default_stats
                for key in loaded_stats:
                    if key in self.stats:
                        self.stats[key] = loaded_stats[key]
            else:
                self.stats = default_stats
        else:
            self.stats = default_stats

    def get_last_call_info(self) -> dict:
        """Get metadata from the most recent AI call."""
        return self._last_call_info.copy() if hasattr(self, '_last_call_info') else {}

    def _record_stats(self, call_type: str, ms: int, token_stats: dict) -> None:
        """Record stats for a call type and persist to storage."""
        self._last_call_info = {
            'model_name': self.model_name,
            'model_ms': ms,
            'model_tokens': token_stats.get('total_tokens', 0),
        }
        if call_type not in self.stats:
            self.stats[call_type] = {'calls': 0, 'total_ms': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        self.stats[call_type]['calls'] += 1
        self.stats[call_type]['total_ms'] += ms
        self.stats[call_type]['prompt_tokens'] += token_stats.get('prompt_tokens', 0)
        self.stats[call_type]['completion_tokens'] += token_stats.get('completion_tokens', 0)
        self.stats[call_type]['total_tokens'] += token_stats.get('total_tokens', 0)

        # Persist to storage
        if self.storage:
            try:
                self.storage.save_api_stats(self.provider_name, self.stats)
            except Exception as e:
                logger.warning(f"Failed to persist API stats: {e}")

    def get_stats(self) -> dict:
        """Get current stats with computed averages."""
        result = {}
        totals = {'calls': 0, 'total_ms': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

        for call_type, stats in self.stats.items():
            calls = stats['calls']
            result[call_type] = {
                **stats,
                'avg_ms': round(stats['total_ms'] / calls, 1) if calls > 0 else 0,
                'avg_tokens': round(stats['total_tokens'] / calls, 1) if calls > 0 else 0
            }
            for key in totals:
                totals[key] += stats[key]

        calls = totals['calls']
        result['total'] = {
            **totals,
            'avg_ms': round(totals['total_ms'] / calls, 1) if calls > 0 else 0,
            'avg_tokens': round(totals['total_tokens'] / calls, 1) if calls > 0 else 0
        }
        return result

    def _execute_chat(self, prompt: str) -> tuple[str, int, dict]:
        """Execute chat and return (text, ms, token_stats)."""
        start_time = time.time()
        response = self.chat.send_message(prompt)
        end_time = time.time()
        ms = int((end_time - start_time) * 1000)

        # Extract token stats from usage_metadata
        token_stats = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            metadata = response.usage_metadata
            token_stats['prompt_tokens'] = getattr(metadata, 'prompt_token_count', 0)
            token_stats['completion_tokens'] = getattr(metadata, 'candidates_token_count', 0)
            token_stats['total_tokens'] = getattr(metadata, 'total_token_count', 0)

        return (response.text, ms, token_stats)

    def _sanitize_judgement(self, judgement: str) -> str:
        s = judgement[judgement.find('{'):judgement.rfind('}')+1]
        s = s.replace('false', 'False')
        s = s.replace('true', 'True')
        return s

    def _get_difficulty_description(self, difficulty: int, lang_name: str, is_reverse: bool = False) -> str:
        """Generate detailed difficulty instructions based on level."""
        # Vocabulary and grammar constraints by level
        level_specs = {
            0: {
                'vocab_size': 50,
                'vocab_desc': 'only the 50 most basic words (I, you, is, the, a, cat, dog, big, small, good, bad, food, water, house, red, blue, yes, no, hello, thank you)',
                'grammar': 'present tense only, simple "X is Y" or "I verb" structures',
                'sentence_words': '2-4',
                'examples': '"The cat is big.", "I like food.", "The house is red."'
            },
            1: {
                'vocab_size': 100,
                'vocab_desc': 'only the 100 most common everyday words (basic nouns: cat, dog, house, food, water, book, car, door, table, chair; basic verbs: is, have, go, eat, drink, see, want, like; basic adjectives: big, small, good, bad, new, old, hot, cold)',
                'grammar': 'present tense only, simple subject-verb-object sentences',
                'sentence_words': '3-6',
                'examples': '"The dog eats food.", "I have a big house.", "She likes the red car."'
            },
            2: {
                'vocab_size': 200,
                'vocab_desc': 'the 200 most common words (expand to include: family words, numbers 1-20, days of week, common foods, basic emotions)',
                'grammar': 'present tense, simple past tense for common verbs (was, had, went)',
                'sentence_words': '4-8',
                'examples': '"Yesterday I went to the store.", "My brother has three cats.", "The food was good."'
            },
            3: {
                'vocab_size': 350,
                'vocab_desc': 'the 350 most common words (add: more verbs, household items, weather, body parts, clothing)',
                'grammar': 'present and past tense, basic questions, simple negations',
                'sentence_words': '5-10',
                'examples': '"Did you see the movie yesterday?", "I don\'t have any money.", "The weather is cold today."'
            },
            4: {
                'vocab_size': 500,
                'vocab_desc': 'the 500 most common words (add: professions, places in city, nature words, more adjectives)',
                'grammar': 'present, past, and future tense, compound sentences with "and/but/or"',
                'sentence_words': '6-12',
                'examples': '"I will go to the doctor tomorrow and she will help me.", "The park is beautiful but it was closed."'
            },
            5: {
                'vocab_size': 750,
                'vocab_desc': 'the 750 most common words (add: abstract nouns, more descriptive adjectives, adverbs)',
                'grammar': 'all basic tenses, relative clauses (who, which, that), common expressions',
                'sentence_words': '8-15',
                'examples': '"The woman who lives next door has a garden that is very beautiful."'
            },
            6: {
                'vocab_size': 1000,
                'vocab_desc': 'the 1000 most common words (intermediate vocabulary)',
                'grammar': 'conditional sentences (if...then), passive voice basics, reported speech',
                'sentence_words': '10-18',
                'examples': '"If it rains tomorrow, we would stay at home.", "The book was written by a famous author."'
            },
            7: {
                'vocab_size': 1500,
                'vocab_desc': 'the 1500 most common words (upper-intermediate vocabulary)',
                'grammar': 'subjunctive mood, complex conditionals, varied sentence structures',
                'sentence_words': '10-20',
                'examples': '"I wish I had studied more when I was younger.", "Had I known earlier, I would have helped."'
            },
            8: {
                'vocab_size': 2500,
                'vocab_desc': 'the 2500 most common words (advanced vocabulary, some literary/formal words)',
                'grammar': 'all tenses and moods, complex clauses, nuanced expressions',
                'sentence_words': '12-22',
                'examples': 'Complex narrative sentences with multiple clauses and sophisticated vocabulary.'
            },
            9: {
                'vocab_size': 4000,
                'vocab_desc': 'the 4000 most common words (near-native vocabulary including idioms)',
                'grammar': 'native-level grammar, idiomatic expressions, colloquialisms',
                'sentence_words': '12-25',
                'examples': 'Sentences with idioms, cultural references, and nuanced meanings.'
            },
            10: {
                'vocab_size': 'unlimited',
                'vocab_desc': 'full native vocabulary including literary, technical, and archaic words',
                'grammar': 'full native complexity, literary devices, regional expressions',
                'sentence_words': '15-30',
                'examples': 'Literary prose with sophisticated vocabulary and complex grammatical structures.'
            }
        }

        spec = level_specs.get(difficulty, level_specs[5])  # Default to level 5 if out of range

        if is_reverse:
            return f"""
            - Difficulty level: {difficulty} out of {MAX_DIFFICULTY}
              The student will translate these English sentences into {lang_name}.
              The difficulty refers to the {lang_name} translation complexity.

              VOCABULARY CONSTRAINTS (CRITICAL - follow strictly):
              - Use ONLY {spec['vocab_desc']}
              - Do NOT use uncommon words like "meadow", "brook", "trail", "glisten", "pond"
              - Prefer concrete, everyday nouns over abstract or literary words

              GRAMMAR CONSTRAINTS:
              - Grammar allowed: {spec['grammar']}
              - Do NOT use grammar structures beyond this level

              SENTENCE LENGTH: {spec['sentence_words']} words per sentence

              EXAMPLES of appropriate sentences: {spec['examples']}"""
        else:
            return f"""
            - Difficulty level: {difficulty} out of {MAX_DIFFICULTY}

              VOCABULARY CONSTRAINTS (CRITICAL - follow strictly):
              - Use ONLY {spec['vocab_desc']}
              - Do NOT use uncommon or literary vocabulary beyond this level
              - Prefer concrete, everyday words over abstract concepts

              GRAMMAR CONSTRAINTS:
              - Grammar allowed: {spec['grammar']}
              - Do NOT use grammar structures beyond this level

              SENTENCE LENGTH: {spec['sentence_words']} words per sentence

              EXAMPLES of appropriate sentences: {spec['examples']}"""

    def generate_story(self, correct_words: list, difficulty: int, direction: str = 'normal', language_info: dict = None) -> tuple[str, int]:
        lang = language_info or _DEFAULT_LANGUAGE_INFO
        lang_name = lang['english_name']
        script = lang.get('script', 'latin')
        avoided_words = correct_words[-100:] if len(correct_words) > 100 else correct_words

        # Generate random elements for story diversity
        seed = int(time.time() * 1000) % 100000
        number = random.randint(0, 10)
        day = random.choice(DAYS_OF_WEEK)
        season = random.choice(SEASONS)
        letter = random.choice("ABCDEFGHJLMNPRST")

        script_instruction = ""
        if script == 'cyrillic':
            script_instruction = f"\n            - IMPORTANT: Write entirely in Cyrillic script. All {lang_name} text must use Cyrillic letters."

        # Swap language based on direction
        if direction == 'reverse':
            story_language = 'English'
            target_language = lang_name
            difficulty_description = self._get_difficulty_description(difficulty, lang_name, is_reverse=True)
        else:
            story_language = lang_name
            target_language = 'English'
            difficulty_description = self._get_difficulty_description(difficulty, lang_name, is_reverse=False)

        prompt = f"""
            Story ID: {seed}

            Write a short, engaging story in {story_language} with approximately {STORY_SENTENCE_COUNT} sentences.

            MANDATORY elements (you MUST include ALL of these):
            - The number {number} appears meaningfully in the story
            - The story takes place on a {day} in {season}
            - The first sentence must include a noun that starts with the letter "{letter}"

            Requirements:
            {difficulty_description}
            - Try to avoid using these nouns (the student already knows them): {', '.join(avoided_words) if avoided_words else 'none'}{script_instruction}

            Write only the story text, no titles, no translations, no explanations.
            Each sentence should end with proper punctuation (. ! ?).
        """
        text, ms, token_stats = self._execute_chat(prompt)
        self._record_stats('story', ms, token_stats)
        return (text, ms)

    def validate_translation(self, sentence: str, translation: str, story_context: str = None, direction: str = 'normal', language_info: dict = None) -> tuple[dict, int, str]:
        lang = language_info or _DEFAULT_LANGUAGE_INFO
        lang_name = lang['english_name']
        context_block = ""
        if story_context:
            context_block = f"""
            STORY CONTEXT (use this to resolve ambiguities like pronouns, possessives, and references):
            "{story_context}"
            """

        # Swap source/target language based on direction
        if direction == 'reverse':
            source_language = 'English'
            target_language = lang_name
        else:
            source_language = lang_name
            target_language = 'English'

        prompt = f"""
            You are evaluating a student's {target_language} translation of a {source_language} sentence.
            {context_block}
            {source_language} Sentence: "{sentence}"
            Student's Translation: "{translation}"

            STEP 1 - WORD-BY-WORD MAPPING (do this carefully):
            For EACH content word in the {source_language} sentence:
            a) List the correct {target_language} translation(s)
            b) Find what {target_language} word the student actually used for it
            c) Determine if they match

            Example analysis:
            - "escuela" → correct: "school" → student used: "stairs" → MISMATCH (mistranslated)
            - "montañas" → correct: "mountains" → student used: "montanas" → MISMATCH (not translated, just copied)
            - "caminan" → correct: "walk/walking" → student used: "walk" → MATCH

            STEP 2 - EVALUATION RULES:
            1. A word is CORRECTLY translated if the student used a valid {target_language} equivalent.
               Multiple {target_language} words can be valid translations (e.g., "vista" can be "view", "sight", "vision").
            2. IDIOMATIC EXPRESSIONS: Some {source_language} phrases have idiomatic meanings that differ from
               literal word-by-word translation. The student should be credited for correct idiomatic translations.
               Examples:
               - "cumplir X años" or "celebrar X años" = "to turn X years old" or "celebrate X birthday"
                 (NOT literally "fulfill X years" or "celebrate X years")
               - "tener X años" = "to be X years old" (NOT "to have X years")
               - "hacer calor/frío" = "to be hot/cold" (NOT "to make heat/cold")
               - "cerca de/del" = "near the" or "close to the" (NOT "near of the")
               - "lejos de" = "far from" (NOT "far of")
               - "antes de" = "before" (NOT "before of")
               - "después de" = "after" (NOT "after of")
               When a student correctly translates an idiom, mark all component words as correct.
            3. A word is INCORRECTLY translated if:
               - Left in {source_language} (not translated at all, e.g., "casa" instead of "house")
               - Copied with accents removed (e.g., "montanas" instead of "mountains" for "montañas",
                 "manana" instead of "tomorrow" for "mañana") - this is NOT a translation
               - Translated to a wrong meaning
               - Omitted entirely
            4. Do NOT penalize the student for:
               - Using synonyms (these are valid translations)
               - Minor word order differences that don't change meaning
               - Using articles (a/an/the) flexibly
               - Capitalization differences (e.g., "monday" instead of "Monday", or not capitalizing the first word)
               - Missing or different punctuation (periods, commas, exclamation marks)
               - Missing accents/diacritics in the {target_language} translation when it does not change
                 the meaning of the word (e.g., "tambien" for "también", "cafe" for "café",
                 "facil" for "fácil" are all acceptable). Only penalize missing accents when
                 they change the word's meaning (e.g., "el" vs "él", "tu" vs "tú", "si" vs "sí").
               - Reflexive pronouns (se, me, te) that are naturally absorbed into English phrasal verbs
                 (e.g., "se pone" → "puts on" is correct, don't require "himself/herself")
               - Prepositions or function words absorbed into another translated phrase
                 (e.g., "del" in "cerca del radiador" → "near the radiator" — the "de" is absorbed
                 into "near" and "el" into "the"; do NOT require a separate "of the")
               - Synonyms that capture the same meaning even if not the most literal translation
                 (e.g., "heater" for "radiador", "big" for "grande", "scared" for "asustado")
               - Grammar elements that have no direct equivalent when the meaning is preserved
               - Omitting subject pronouns (I, he, she, it, they, etc.) in pro-drop languages where
                 the subject is implied or unnecessary (e.g., "It is winter" → "Zima je" is correct,
                 "I eat" → "Como" is correct — do NOT require an explicit pronoun)
               - Different but valid word order (e.g., "Zima je" vs "Je zima" — both valid)
               - Providing multiple valid translations separated by commas or slashes
                 (e.g., "fresh, cool" for "fresca" — if all variants are correct translations,
                 score should be 100)
            5. Only proper nouns (names of people, places, brands) may remain in original form.
               Common nouns MUST be translated.
            6. If the translation is semantically correct with all words properly translated, the score should be 100.
            7. IMPORTANT: If the student's translation conveys the complete meaning naturally in {target_language},
               give 100. Do not deduct points for stylistic differences or implied grammar.
            8. CONTEXT-DEPENDENT WORDS: Words like "su" (his/her/your/their) or "este/esta" must be
               evaluated based on the story context. If the story is about a male character (e.g., Mateo),
               then "su hermana" should be "his sister", not "her sister". Accept any translation that
               is valid given the story context. Do NOT penalize for choosing one valid interpretation
               when the context supports it.

            STEP 3 - RESPOND with a Python dictionary:

            'vocabulary_breakdown': A list of lists, one for each meaningful word/phrase:
              [0] The {source_language} word/phrase from the original sentence (MUST be in {source_language})
              [1] The correct {target_language} translation(s) for this word in context (MUST be in {target_language})
              IMPORTANT: The order is ALWAYS [{source_language} word, {target_language} translation].
              For example, if source is English and target is Spanish: ["swim", "nadan"] NOT ["nadan", "swim"].
              [2] Part of speech (noun, verb, adjective, etc.)
              [3] Boolean: True ONLY if the student's translation contains a valid {target_language} equivalent.
                  False if: the word was mistranslated,
                  left untranslated, or omitted entirely.
                  IMPORTANT: Check the actual {target_language} word the student used, not just sentence meaning.

            'correct_translation': A proper {target_language} translation of the sentence.
              If the student's translation is correct, you may use their translation.
              If the score is less than 100, show a translation that demonstrates what was missed or wrong.

            'score': Integer 1-100. Base it on:
              - What percentage of words were correctly translated
              - Deduct heavily for untranslated {source_language} words left as-is
              - Deduct for wrong meanings
              - Minor deductions for grammar issues

            'evaluation': Brief explanation. Be ACCURATE:
              - Do not claim a word was "added" if it's a valid translation or part of an idiom
              - Only mention actual errors, not valid idiomatic translations

            Return ONLY the dictionary, no other text, no markdown formatting.
        """
        response, ms, token_stats = self._execute_chat(prompt)
        self._record_stats('validate', ms, token_stats)
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

            # Filter out malformed vocabulary_breakdown entries (each must be a list of 4+)
            if 'vocabulary_breakdown' in judgement:
                judgement['vocabulary_breakdown'] = [
                    entry for entry in judgement['vocabulary_breakdown']
                    if isinstance(entry, (list, tuple)) and len(entry) >= 4
                ]

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
        return (judgement, ms, response)

    def get_hint(self, sentence: str, correct_words: list, direction: str = 'normal', partial_translation: str = '', language_info: dict = None) -> dict | None:
        lang = language_info or _DEFAULT_LANGUAGE_INFO
        lang_name = lang['english_name']
        partial_section = ''
        if partial_translation:
            partial_section = f"""
            The student's current partial translation: "{partial_translation}"
            Analyze which words the student has already correctly translated in their partial attempt.
            Do NOT hint at words they have already correctly translated — focus ONLY on words they are still missing or got wrong.
            IMPORTANT: You MUST provide hints for any missing/untranslated words regardless of how common they are. Even simple words like "too", "also", "very", "well" etc. should be hinted if they are missing from the partial translation.
"""

        if direction == 'reverse':
            # Reverse mode: English sentence, provide target language translations
            prompt = f"""
            Analyze this English sentence and provide a hint for translating to {lang_name}:

            Sentence: "{sentence}"

            Already known words (do NOT include these): {', '.join(correct_words[-50:]) if correct_words else 'none'}
            {partial_section}
            Find a noun, verb, adjective, and adverb from the sentence that the student needs help with.
            If a partial translation is provided, prioritize words that are still missing from the translation.
            Otherwise, prioritize less common vocabulary that a student would most likely need help with.
            Provide their {lang_name} translations.

            Respond with ONLY a Python dictionary in this exact format:
            {{
                'noun': ['english_noun', '{lang_name.lower()}_translation'],
                'verb': ['english_verb', '{lang_name.lower()}_translation'],
                'adjective': ['english_adjective', '{lang_name.lower()}_translation'],
                'adverb': ['english_adverb', '{lang_name.lower()}_translation']
            }}

            If any part of speech is not available (all are known or not present), use null for that entry:
            {{'noun': null, 'verb': ['english_verb', '{lang_name.lower()}_translation'], 'adjective': null, 'adverb': null}}

            Return ONLY the dictionary, no other text.
        """
        else:
            prompt = f"""
            Analyze this {lang_name} sentence and provide a hint for translation:

            Sentence: "{sentence}"

            Already known words (do NOT include these): {', '.join(correct_words[-50:]) if correct_words else 'none'}
            {partial_section}
            Find a noun, verb, adjective, and adverb from the sentence that the student needs help with.
            If a partial translation is provided, prioritize words that are still missing from the translation.
            Otherwise, prioritize less common vocabulary that a student would most likely need help with.
            Provide their English translations.

            Respond with ONLY a Python dictionary in this exact format:
            {{
                'noun': ['{lang_name.lower()}_noun', 'english_translation'],
                'verb': ['{lang_name.lower()}_verb', 'english_translation'],
                'adjective': ['{lang_name.lower()}_adjective', 'english_translation'],
                'adverb': ['{lang_name.lower()}_adverb', 'english_translation']
            }}

            If any part of speech is not available (all are known or not present), use null for that entry:
            {{'noun': null, 'verb': ['{lang_name.lower()}_verb', 'english_translation'], 'adjective': null, 'adverb': null}}

            Return ONLY the dictionary, no other text.
        """
        response, ms, token_stats = self._execute_chat(prompt)
        self._record_stats('hint', ms, token_stats)
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

    def get_word_translation(self, word: str, language_info: dict = None) -> dict | None:
        """Get translation and type for a word in the target language.
        Returns dict with 'translation' and 'type' or None on error."""
        lang = language_info or _DEFAULT_LANGUAGE_INFO
        lang_name = lang['english_name']
        prompt = f"""
            Translate this {lang_name} word to English:

            Word: "{word}"

            Provide:
            1. The English translation(s) - if multiple common translations exist, separate with commas
            2. The part of speech (noun, verb, adjective, adverb, etc.)

            Respond with ONLY a Python dictionary in this exact format:
            {{'translation': 'english translation(s)', 'type': 'part of speech'}}

            Examples:
            - "casa" -> {{'translation': 'house, home', 'type': 'noun'}}
            - "correr" -> {{'translation': 'to run', 'type': 'verb'}}
            - "rápido" -> {{'translation': 'fast, quick', 'type': 'adjective'}}

            Return ONLY the dictionary, no other text.
        """
        response, ms, token_stats = self._execute_chat(prompt)
        self._record_stats('word_translation', ms, token_stats)
        raw_response = response
        try:
            response = response.strip()
            response = response.replace('```python', '').replace('```', '')
            result = eval(response)

            if not isinstance(result, dict):
                logger.warning(f"Word translation response is not a dict: {type(result)}")
                return None

            if 'translation' not in result or 'type' not in result:
                logger.warning(f"Word translation missing required fields: {result}")
                return None

            return result
        except Exception as e:
            logger.error(f"Failed to parse word translation: {e}")
            logger.error(f"Raw response:\n{raw_response}")
            return None

    def validate_verb_translation(self, conjugated_form: str, base_verb: str,
                                    correct_translation: str, student_answer: str,
                                    language_info: dict = None) -> dict:
        """Check if student's answer is a valid translation of the verb, any tense accepted.
        Returns dict with 'translation_correct' (bool) and 'explanation' (str)."""
        lang = language_info or _DEFAULT_LANGUAGE_INFO
        lang_name = lang['english_name']
        prompt = f"""
            A student is translating the {lang_name} verb "{conjugated_form}" (infinitive: {base_verb}).
            The expected translation is: "{correct_translation}"
            The student answered: "{student_answer}"

            Is the student's answer a valid English translation of this verb?
            Accept ANY English tense form as correct. For example:
            - "fly", "flew", "flies", "flying", "flown" are ALL correct for the verb "volar"
            - "run", "ran", "runs", "running" are ALL correct for "correr"
            - But "eat" would be WRONG for "correr"

            Respond with ONLY a Python dictionary:
            {{'correct': True/False, 'explanation': 'brief reason'}}
        """
        response, ms, token_stats = self._execute_chat(prompt)
        self._record_stats('verb_analysis', ms, token_stats)
        try:
            response = response.strip().replace('```python', '').replace('```', '')
            response = response.replace('true', 'True').replace('false', 'False')
            result = eval(response)
            if isinstance(result, dict) and 'correct' in result:
                return result
        except Exception as e:
            logger.error(f"Failed to parse verb validation: {e}")
        return {'correct': False, 'explanation': 'Could not evaluate'}

    def analyze_verb_conjugation(self, conjugated_form: str, language_info: dict = None) -> dict | None:
        """Analyze a conjugated verb form.
        Returns dict with base_verb, tense, translation, person or None on error."""
        lang = language_info or _DEFAULT_LANGUAGE_INFO
        lang_name = lang['english_name']
        tenses = lang.get('tenses', ['present', 'preterite', 'imperfect', 'future', 'conditional', 'subjunctive'])
        tense_list_str = ', '.join(tenses)

        prompt = f"""
            Analyze this conjugated {lang_name} verb:

            Verb form: "{conjugated_form}"

            Identify:
            1. The infinitive (base form) of the verb
            2. The tense - must be one of: {tense_list_str}
            3. The English translation of this specific conjugated form
            4. The person - must be one of: first_singular, second_singular, third_singular, first_plural, second_plural, third_plural

            Respond with ONLY a Python dictionary in this exact format:
            {{'base_verb': 'infinitive', 'tense': 'tense_name', 'translation': 'english', 'person': 'person_code'}}

            Return ONLY the dictionary, no other text.
        """
        response, ms, token_stats = self._execute_chat(prompt)
        self._record_stats('verb_analysis', ms, token_stats)
        raw_response = response
        try:
            response = response.strip()
            response = response.replace('```python', '').replace('```', '')
            result = eval(response)

            if not isinstance(result, dict):
                logger.warning(f"Verb conjugation response is not a dict: {type(result)}")
                return None

            required_fields = ['base_verb', 'tense', 'translation', 'person']
            for field in required_fields:
                if field not in result:
                    logger.warning(f"Verb conjugation missing required field: {field}")
                    return None

            # Validate tense is one of expected values
            valid_tenses = tenses
            if result['tense'] not in valid_tenses:
                logger.warning(f"Invalid tense '{result['tense']}', expected one of {valid_tenses}")
                # Try to normalize common variations
                tense_map = {
                    'past': 'preterite',
                    'simple_past': 'preterite',
                    'past_simple': 'preterite',
                    'present_subjunctive': 'subjunctive',
                    'imperfecto': 'imperfect',
                    'preterito': 'preterite',
                    'futuro': 'future',
                    'condicional': 'conditional',
                    'presente': 'present',
                }
                result['tense'] = tense_map.get(result['tense'].lower(), result['tense'])

            return result
        except Exception as e:
            logger.error(f"Failed to parse verb conjugation: {e}")
            logger.error(f"Raw response:\n{raw_response}")
            return None

    def generate_synonym_antonym(self, word: str, language_info: dict = None) -> dict | None:
        """Generate a synonym and antonym for a word in the target language.
        Returns dict with 'synonym' and 'antonym' (antonym may be None) or None on error."""
        lang = language_info or _DEFAULT_LANGUAGE_INFO
        lang_name = lang['english_name']
        prompt = f"""
            Given the {lang_name} word '{word}', provide a common synonym and a common antonym
            (if a natural antonym exists).

            Return ONLY a Python dictionary in this exact format:
            {{'synonym': 'synonym_word', 'antonym': 'antonym_word'}}

            Use None if no natural synonym exists.
            Use None if no natural antonym exists.

            Examples:
            - "grande" -> {{'synonym': 'enorme', 'antonym': 'pequeño'}}
            - "casa" -> {{'synonym': 'hogar', 'antonym': None}}
            - "rápido" -> {{'synonym': 'veloz', 'antonym': 'lento'}}

            Return ONLY the dictionary, no other text.
        """
        try:
            response, ms, token_stats = self._execute_chat(prompt)
            self._record_stats('synonym_challenge', ms, token_stats)
            response = response.strip()
            response = response.replace('```python', '').replace('```', '')
            response = response.replace('null', 'None')
            result = eval(response)
            if isinstance(result, dict):
                return result
            return None
        except Exception as e:
            logger.error(f"Failed to generate synonym/antonym for '{word}': {e}")
            return None

    def validate_synonym_antonym(self, word: str, challenge_type: str, expected: str,
                                  student_answer: str, language_info: dict = None) -> dict:
        """Validate if the student's answer is a valid synonym/antonym.
        Returns dict with 'correct' (bool) and 'explanation' (str)."""
        lang = language_info or _DEFAULT_LANGUAGE_INFO
        lang_name = lang['english_name']
        type_label = 'synonym' if challenge_type == 'SYN' else 'antonym'
        prompt = f"""
            A student was asked for a {type_label} of the {lang_name} word "{word}".
            The expected answer was: "{expected}"
            The student answered: "{student_answer}"

            Is the student's answer a valid {type_label} of "{word}" in {lang_name}?
            Accept common, natural {type_label}s even if they differ from the expected answer.

            Respond with ONLY a Python dictionary:
            {{'correct': True/False, 'explanation': 'brief reason'}}
        """
        try:
            response, ms, token_stats = self._execute_chat(prompt)
            self._record_stats('synonym_challenge', ms, token_stats)
            response = response.strip().replace('```python', '').replace('```', '')
            response = response.replace('true', 'True').replace('false', 'False')
            result = eval(response)
            if isinstance(result, dict) and 'correct' in result:
                return result
        except Exception as e:
            logger.error(f"Failed to validate synonym/antonym: {e}")
        return {'correct': False, 'explanation': 'Could not evaluate'}

    def generate_conjugation_rules(self, tense: str, language_info: dict = None) -> str | None:
        """Generate a concise conjugation rules summary for a tense in the target language.
        Returns plain text string or None on error."""
        lang = language_info or _DEFAULT_LANGUAGE_INFO
        lang_name = lang['english_name']

        prompt = f"""
            Write a concise conjugation cheat-sheet for the {tense} tense in {lang_name}.

            Requirements:
            - Show the regular verb endings/patterns grouped by verb class (e.g. -ar, -er, -ir for Spanish)
            - For each pattern, show all person forms with a common example verb
            - Keep it short: just the conjugation table, no lengthy explanations
            - Use plain text, no markdown

            Example format for Spanish present tense:
            -ar (hablar): yo hablo, tú hablas, él/ella habla, nosotros hablamos, ellos/ellas hablan
            -er (comer): yo como, tú comes, él/ella come, nosotros comemos, ellos/ellas comen
            -ir (vivir): yo vivo, tú vives, él/ella vive, nosotros vivimos, ellos/ellas viven

            Adapt the format to {lang_name} grammar (e.g. different verb classes, different pronoun system).
            Return ONLY the conjugation patterns, no other text.
        """
        try:
            response, ms, token_stats = self._execute_chat(prompt)
            self._record_stats('verb_hint', ms, token_stats)
            return response.strip() if response else None
        except Exception as e:
            logger.error(f"Failed to generate conjugation rules: {e}")
            return None

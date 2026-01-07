"""Console UI for tongue application."""

from core.config import (
    LANGUAGE, MAX_DIFFICULTY,
    ADVANCE_SCORE_THRESHOLD, ADVANCE_REQUIRED_GOOD, ADVANCE_WINDOW_SIZE,
    DEMOTE_SCORE_THRESHOLD, DEMOTE_REQUIRED_POOR
)
from core.utils import split_into_sentences
from cli.api_client import TongueAPIClient


class ConsoleUI:
    """Console user interface for tongue application."""

    def __init__(self, client: TongueAPIClient):
        self.client = client

    def print_story_with_highlight(self, story: str, current_sentence: str, difficulty: int):
        """Print the story with the current sentence highlighted."""
        if not story:
            return
        print('\n' + '=' * 60)
        print(f'STORY (Level {difficulty})')
        print('=' * 60)
        all_sentences = split_into_sentences(story)
        for sentence in all_sentences:
            if sentence == current_sentence:
                print(f'\n  >>> {sentence} <<<\n')
            else:
                print(f'  {sentence}')
        print('=' * 60)

    def print_evaluation(self, result: dict, sentence: str, translation: str):
        """Print evaluation results."""
        print('-' * 40)
        print(f'Input sentence in {LANGUAGE}: {sentence}')
        print(f'Your translation: {translation}')
        print(f'Judgement time: {result["judge_ms"]}ms')
        print(f"Correct translation: {result['correct_translation']}")
        print(f"Score: {result['score']}")
        if result['score'] < 100:
            print(f"Reason: {result['evaluation']}")
        print('-' * 40)

        if result['level_changed']:
            if result['change_type'] == 'advanced':
                print(f"\n*** LEVEL UP! Now at level {result['new_level']} ***\n")
            elif result['change_type'] == 'demoted':
                print(f"\n*** Level down. Now at level {result['new_level']} ***\n")

    def print_previous_evaluation(self, prev: dict):
        """Print previous round's evaluation."""
        print('-' * 40)
        print(f'Input sentence in {LANGUAGE}: {prev["sentence"]}')
        print(f'Your translation: {prev["translation"]}')
        print(f'Judgement time: {prev["judge_ms"]}ms')
        print(f"Correct translation: {prev['correct_translation']}")
        print(f"Score: {prev['score']}")
        if prev.get('evaluation'):
            print(f"Reason: {prev['evaluation']}")
        print('-' * 40)

        if prev.get('level_changed'):
            print("\n*** Level changed! ***\n")

    def print_status(self, status: dict):
        """Print detailed status."""
        print('\n' + '=' * 50)
        print('STATUS SUMMARY')
        print('=' * 50)
        print(f'\nLanguage: {status["language"]}')
        print(f'Current level: {status["difficulty"]}/{status["max_difficulty"]}')
        print(f'Total sentences completed: {status["total_completed"]}')

        if status['story_sentences_remaining'] > 0:
            print(f'Current story: {status["story_sentences_remaining"]} sentences remaining')
        else:
            print('Current story: completed (new story will generate next)')

        if status['level_scores']:
            avg = sum(status['level_scores']) / len(status['level_scores'])
            print(f'\nLevel {status["difficulty"]} stats (last {len(status["level_scores"])} attempts):')
            print(f'  Average score: {avg:.1f}')
            print(f'  Good scores (>={ADVANCE_SCORE_THRESHOLD}): {status["good_score_count"]}/{ADVANCE_REQUIRED_GOOD} needed to advance')
            print(f'  Poor scores (<{DEMOTE_SCORE_THRESHOLD}): {status["poor_score_count"]}/{DEMOTE_REQUIRED_POOR} triggers demotion')
        else:
            print(f'\nNo scores yet at level {status["difficulty"]}')

        print(f'\nMastered words: {status["mastered_words_count"]}')
        if status['recent_mastered']:
            print(f'  Recent: {", ".join(status["recent_mastered"])}')

        print(f'\nWords to practice: {status["missed_words_count"]}')
        print('\n' + '=' * 50 + '\n')

    def print_hint(self, hint: dict):
        """Print a hint."""
        if hint.get('noun') or hint.get('verb') or hint.get('adjective'):
            print('\n--- HINT ---')
            if hint.get('noun'):
                print(f"  Noun: {hint['noun'][0]} = {hint['noun'][1]}")
            if hint.get('verb'):
                print(f"  Verb: {hint['verb'][0]} = {hint['verb'][1]}")
            if hint.get('adjective'):
                print(f"  Adjective: {hint['adjective'][0]} = {hint['adjective'][1]}")
            print('------------')
        else:
            print('Could not generate hint.')

    def print_progress_bar(self, status: dict):
        """Print progress information."""
        print('=' * 40)
        print(f'Total completed: {status["total_completed"]} | {status["progress_display"]}')
        if status['level_scores']:
            avg = sum(status['level_scores']) / len(status['level_scores'])
            print(f'Recent average: {avg:.1f} (need {ADVANCE_SCORE_THRESHOLD}+ for good, <{DEMOTE_SCORE_THRESHOLD} is poor)')
            if status['difficulty'] > 1 and status['poor_score_count'] >= 2:
                print(f'Warning: {status["poor_score_count"]}/{DEMOTE_REQUIRED_POOR} poor scores - risk of demotion!')
        print('=' * 40)

    def run(self):
        """Run the main application loop."""
        # Check server connection
        try:
            health = self.client.health_check()
            print(f"Connected to tongue server ({health['service']})")
        except Exception as e:
            print(f"Error: Cannot connect to server at {self.client.base_url}")
            print(f"Make sure the server is running: python run_server.py")
            return

        # Get initial status
        status = self.client.get_status()
        print(f"Restored: {status['total_completed']} completed, {status['progress_display']}")

        print(f'\nStarting {LANGUAGE} translation practice!')
        print(f'Level up: Score {ADVANCE_SCORE_THRESHOLD}+ on {ADVANCE_REQUIRED_GOOD}/{ADVANCE_WINDOW_SIZE} recent attempts')
        print(f'Level down: Score <{DEMOTE_SCORE_THRESHOLD} on {DEMOTE_REQUIRED_POOR}/{ADVANCE_WINDOW_SIZE} recent attempts')
        print('Commands: "hint" for help, "status" for progress, "exit" to quit\n')

        while True:
            # Get next sentence
            try:
                data = self.client.get_next_sentence()
            except Exception as e:
                print(f"Error getting next sentence: {e}")
                continue

            sentence = data['sentence']
            story = data['story']
            difficulty = data['difficulty']

            # 1. Display story
            self.print_story_with_highlight(story, sentence, difficulty)

            # 2. Display previous evaluation if any
            if data['has_previous_evaluation'] and data['previous_evaluation']:
                self.print_previous_evaluation(data['previous_evaluation'])
                # Print progress
                status = self.client.get_status()
                self.print_progress_bar(status)

            # 3. Display new task
            print(f"\n{data['progress_display']}")
            print(f"\n>>> {sentence}")

            # Get user input
            translation = ''
            while not translation:
                user_input = input('==> ').strip()

                if user_input.lower() == 'exit':
                    print('Goodbye!')
                    return

                elif user_input.lower() == 'hint':
                    print('Getting hint...')
                    try:
                        hint = self.client.get_hint(sentence)
                        self.print_hint(hint)
                    except Exception as e:
                        print(f"Error getting hint: {e}")
                    print(f"\n>>> {sentence}")

                elif user_input.lower() == 'status':
                    try:
                        status = self.client.get_status()
                        self.print_status(status)
                    except Exception as e:
                        print(f"Error getting status: {e}")
                    print(f"\n>>> {sentence}")

                elif user_input == '':
                    self.print_story_with_highlight(story, sentence, difficulty)
                    print(f"\n>>> {sentence}")

                else:
                    translation = user_input

            # Submit translation
            print('Validating translation...')
            try:
                result = self.client.submit_translation(sentence, translation)
                self.print_evaluation(result, sentence, translation)

                # Print updated progress
                status = self.client.get_status()
                self.print_progress_bar(status)

            except Exception as e:
                print(f"Error submitting translation: {e}")

"""Configuration constants for tongue application."""

DEFAULT_LANGUAGE = 'es'
MIN_DIFFICULTY = 0
MAX_DIFFICULTY = 10

# Advancement criteria
ADVANCE_WINDOW_SIZE = 10      # Number of recent scores to consider
ADVANCE_SCORE_THRESHOLD = 80  # Minimum score to count as "good"
ADVANCE_REQUIRED_GOOD = 7     # Number of good scores needed to advance

# Demotion criteria
DEMOTE_SCORE_THRESHOLD = 50   # Scores below this count as "poor"
DEMOTE_REQUIRED_POOR = 4      # Number of poor scores to trigger demotion

# Story generation
STORY_SENTENCE_COUNT = 30     # Number of sentences per story

# Challenge selection
CHALLENGE_PROBABILITY = 0.3  # 30% chance of a challenge each turn

# Practice time tracking
PRACTICE_TIME_INACTIVITY_THRESHOLD = 60  # seconds - deltas above this are ignored (user AFK)

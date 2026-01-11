"""Configuration constants for tongue application."""

LANGUAGE = 'Spanish'
MIN_DIFFICULTY = 1
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

"""Utility functions for tongue application."""

import re


def split_into_sentences(text: str) -> list[str]:
    """Split a story into individual sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    cleaned = []
    for s in sentences:
        s = s.strip()
        s = re.sub(r'^\d+[.)]\s*', '', s)
        if s and len(s) > 3:
            cleaned.append(s)
    return cleaned

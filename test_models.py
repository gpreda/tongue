#!/usr/bin/env python3
"""Compare story generation between Gemini models."""

import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from server.gemini_provider import GeminiProvider


def get_api_key():
    """Get API key from env or config file."""
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        config_path = Path.home() / '.config' / 'tongue' / 'config.json'
        if config_path.exists():
            config = json.loads(config_path.read_text())
            api_key = config.get('gemini_api_key')
    return api_key


def main():
    api_key = get_api_key()
    if not api_key:
        print("Error: GEMINI_API_KEY not found")
        return 1

    difficulty = 3
    correct_words = []  # Empty for fresh story

    print("Generating stories at difficulty level 3...\n")
    print("=" * 80)

    # Model 1: Default (gemini-2.0-flash)
    print("\n[MODEL 1: gemini-2.0-flash]\n")
    provider1 = GeminiProvider(api_key, model_name='gemini-2.0-flash')
    story1, ms1 = provider1.generate_story(correct_words, difficulty)
    print(story1)
    print(f"\n⏱ Time: {ms1}ms")
    stats1 = provider1.get_stats()
    print(f"📊 Tokens: {stats1['story']['total_tokens']}")

    print("\n" + "-" * 80)

    # Model 2: gemini-2.5-pro
    print("\n[MODEL 2: gemini-2.5-pro]\n")
    provider2 = GeminiProvider(api_key, model_name='gemini-2.5-pro')
    story2, ms2 = provider2.generate_story(correct_words, difficulty)
    print(story2)
    print(f"\n⏱ Time: {ms2}ms")
    stats2 = provider2.get_stats()
    print(f"📊 Tokens: {stats2['story']['total_tokens']}")

    print("\n" + "=" * 80)
    print("\n[COMPARISON SUMMARY]\n")
    print(f"{'Metric':<20} {'gemini-2.0-flash':<25} {'gemini-2.5-pro':<25}")
    print("-" * 70)
    print(f"{'Time (ms)':<20} {ms1:<25} {ms2:<25}")
    print(f"{'Total tokens':<20} {stats1['story']['total_tokens']:<25} {stats2['story']['total_tokens']:<25}")
    print(f"{'Sentences':<20} {len([s for s in story1.split('.') if s.strip()]):<25} {len([s for s in story2.split('.') if s.strip()]):<25}")

    return 0


if __name__ == '__main__':
    sys.exit(main())

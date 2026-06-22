#!/usr/bin/env python3
"""Cleanup script to re-validate and fix all word translations in user state.

Uses Gemini to validate existing translations and fix incorrect ones.
Processes words in batches for efficiency.

Usage:
    python scripts/cleanup_word_translations.py [--dry-run] [--user USER_ID]
"""

import ast
import json
import os
import sys
import time
import argparse

import psycopg2
import google.generativeai as genai


DB_URL = "postgresql://predator@localhost:5432/tongue"
BATCH_SIZE = 40  # words per Gemini call


def get_api_key():
    key = os.environ.get('GEMINI_API_KEY')
    if key:
        return key
    config_path = os.path.expanduser('~/.config/tongue/config.json')
    try:
        with open(config_path) as f:
            return json.load(f).get('gemini_api_key')
    except FileNotFoundError:
        pass
    raise RuntimeError("No GEMINI_API_KEY found in env or config file")


def batch_validate(model, word_pairs, language='Spanish'):
    """Ask Gemini to validate a batch of word->translation pairs.

    word_pairs: list of (word, current_translation) tuples
    Returns: dict of word -> {'correct_translation': str, 'type': str} for WRONG entries only.
    """
    lines = []
    for i, (word, trans) in enumerate(word_pairs):
        lines.append(f'  {i+1}. {word} -> {trans}')
    pair_list = '\n'.join(lines)

    prompt = f"""
You are validating a dictionary of {language}-to-English word translations.
For each entry below, check if the English translation is a correct dictionary meaning of the {language} word.

Entries:
{pair_list}

For each entry, determine if the current translation is WRONG. A translation is WRONG if:
- It is not a valid English meaning of the {language} word at all (e.g., "ido" -> "5" is WRONG, should be "gone")
- It is empty or blank
- It is a number when the word is not a number
- It is actually the translation in the WRONG direction (e.g., a {language} word stored as the translation for an English word)
- It is a completely different word's meaning (e.g., "De" -> "Suddenly" is WRONG, should be "of, from")

A translation is ACCEPTABLE (not wrong) if:
- It is a valid meaning even if not complete (e.g., "abuela" -> "grandmother" is fine even without "grandma")
- It uses slightly different wording (e.g., "opened" vs "he/she opened" — both fine)
- It gives one valid meaning among several possible ones
- It includes context like "He/She" for conjugated verbs

Respond with ONLY a Python dictionary containing ONLY the entries that need fixing.
Format:
{{
    'word1': {{'correct_translation': 'the correct English meaning(s)', 'type': 'part of speech'}},
    ...
}}

If ALL entries are acceptable, respond with: {{}}
Return ONLY the dictionary, no other text.
"""
    chat = model.start_chat(history=[])
    response = chat.send_message(prompt)
    text = response.text.strip()
    text = text.replace('```python', '').replace('```', '').strip()
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError) as e:
        print(f"  WARNING: Failed to parse batch response: {e}")
        print(f"  Response preview: {text[:300]}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Cleanup word translations')
    parser.add_argument('--dry-run', action='store_true', help='Show changes without applying')
    parser.add_argument('--user', default='gpreda@gmail.com', help='User ID to fix')
    args = parser.parse_args()

    api_key = get_api_key()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3.5-flash')

    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    # Load current words from user state
    cur.execute("SELECT state->'words' FROM user_state WHERE user_id = %s", (args.user,))
    row = cur.fetchone()
    if not row or not row[0]:
        print("No words found for user")
        return

    words_data = row[0] if isinstance(row[0], dict) else json.loads(row[0])
    word_keys = [w for w in words_data.keys() if w.strip()]
    print(f"Total words to validate: {len(word_keys)}")

    # Build word->translation pairs
    word_pairs = []
    for w in word_keys:
        trans = words_data[w].get('translation', '')
        if isinstance(trans, list):
            trans = ', '.join(str(x) for x in trans)
        word_pairs.append((w, trans))

    # Process in batches
    fixes = {}  # word -> {correct_translation, type}
    for i in range(0, len(word_pairs), BATCH_SIZE):
        batch = word_pairs[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(word_pairs) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\nBatch {batch_num}/{total_batches} ({len(batch)} words)...")

        result = batch_validate(model, batch)
        if result is None:
            print("  Batch failed, retrying with smaller batches...")
            # Split into smaller batches of 10
            for j in range(0, len(batch), 10):
                sub_batch = batch[j:j + 10]
                sub_result = batch_validate(model, sub_batch)
                if sub_result:
                    fixes.update(sub_result)
                time.sleep(1)
            continue

        if result:
            for word, info in result.items():
                # Verify the word is in our batch (case-insensitive match)
                matched = None
                for bw, _ in batch:
                    if bw == word or bw.lower() == word.lower():
                        matched = bw
                        break
                if matched:
                    fixes[matched] = info
                    old_trans = words_data[matched].get('translation', '')
                    if isinstance(old_trans, list):
                        old_trans = ', '.join(str(x) for x in old_trans)
                    print(f"  FIX: {matched:25s}: {old_trans!r:30s} -> {info.get('correct_translation', '')!r}")

        # Rate limit
        time.sleep(1)

    # Report changes
    print(f"\n{'=' * 60}")
    print(f"Words that need fixing: {len(fixes)}/{len(word_keys)}")
    print(f"{'=' * 60}")

    for word in sorted(fixes.keys()):
        info = fixes[word]
        old_trans = words_data[word].get('translation', '')
        if isinstance(old_trans, list):
            old_trans = ', '.join(str(x) for x in old_trans)
        old_type = words_data[word].get('type', 'unknown')
        new_trans = info.get('correct_translation', '')
        new_type = info.get('type', old_type)
        print(f"  {word:25s}: {old_trans!r:30s} -> {new_trans!r:30s} (type: {old_type} -> {new_type})")

    if args.dry_run:
        print("\n[DRY RUN] No changes applied.")
        return

    if not fixes:
        print("\nNo fixes needed!")
        return

    # Apply fixes via jsonb_set
    print(f"\nApplying {len(fixes)} fixes...")
    for word, info in fixes.items():
        new_trans = info.get('correct_translation', '')
        new_type = info.get('type', words_data[word].get('type', 'unknown'))

        # Update translation and type in the JSONB words object
        cur.execute("""
            UPDATE user_state
            SET state = jsonb_set(
                jsonb_set(state, ARRAY['words', %s, 'translation'], %s::jsonb),
                ARRAY['words', %s, 'type'], %s::jsonb
            ),
            updated_at = NOW()
            WHERE user_id = %s
        """, (word, json.dumps(new_trans), word, json.dumps(new_type), args.user))

    conn.commit()
    print(f"Done! Fixed {len(fixes)} word translations.")

    # Also clean up the word_translations cache table
    print("\nCleaning word_translations cache table...")
    cur.execute("DELETE FROM word_translations")
    deleted = cur.rowcount
    conn.commit()
    print(f"Cleared {deleted} cached entries (will be re-populated on demand).")

    cur.close()
    conn.close()


if __name__ == '__main__':
    main()

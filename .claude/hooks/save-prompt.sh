#!/bin/bash
# Save every user prompt to a JSONL log for session continuity
INPUT=$(cat)
PROMPT=$(echo "$INPUT" | jq -r '.prompt')
SESSION=$(echo "$INPUT" | jq -r '.session_id')
CWD=$(echo "$INPUT" | jq -r '.cwd')

LOG_DIR="/home/predator/repo/tongue/.claude"
PROMPT_LOG="$LOG_DIR/prompt-history.jsonl"
CURRENT_TASK="$LOG_DIR/current-task.md"

# Append timestamped prompt to history
echo "{\"ts\": \"$(date -Iseconds)\", \"session\": \"$SESSION\", \"prompt\": $(echo "$PROMPT" | jq -Rs .)}" >> "$PROMPT_LOG"

# Keep only the last 200 entries to avoid unbounded growth
if [ $(wc -l < "$PROMPT_LOG") -gt 200 ]; then
    tail -n 200 "$PROMPT_LOG" > "$PROMPT_LOG.tmp" && mv "$PROMPT_LOG.tmp" "$PROMPT_LOG"
fi

exit 0

# Project: Tongue (Language Learning App)

## Session Continuity

When the user says "continue", "continue where we left off", "pick up where we left off", or similar:
1. Read `.claude/current-task.md` for the last task description and status
2. Read `.claude/prompt-history.jsonl` (last ~10 entries) to understand recent conversation flow
3. Resume work from where it was interrupted

**After understanding a new task or at key milestones**, update `.claude/current-task.md` with:
- What the task is
- What has been done so far
- What remains to be done
- Key decisions made

This ensures work survives connection drops.

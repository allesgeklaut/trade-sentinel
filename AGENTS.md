# AGENTS.md

## Commit Policy

NEVER commit changes unless the user explicitly tells you to commit.

## Merge Policy

When merging a feature branch into `main`, create a merge commit
(`git merge --no-ff <branch>`) instead of a fast-forward, so the branch's
history is preserved as a distinct unit.
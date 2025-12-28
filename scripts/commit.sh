#!/bin/bash
# Commit script for sherpa backend
# Usage: ./scripts/commit.sh "commit message"
# Or run without args for interactive mode

set -e

# Ensure we're in the sherpa directory
cd "$(dirname "$0")/.."

echo "=== Sherpa Backend Commit ==="
echo ""

# Show current status
echo "📋 Git Status:"
git status --short
echo ""

# Check if there are changes
if git diff --quiet && git diff --cached --quiet && [ -z "$(git ls-files --others --exclude-standard)" ]; then
    echo "✅ No changes to commit"
    exit 0
fi

# Get commit message
if [ -n "$1" ]; then
    MESSAGE="$1"
else
    echo "📝 Enter commit message (or 'q' to quit):"
    read -r MESSAGE
    if [ "$MESSAGE" = "q" ]; then
        echo "Aborted."
        exit 0
    fi
fi

# Add all changes
echo ""
echo "📦 Staging changes..."
git add -A

# Commit with co-author
echo "💾 Committing..."
git commit -m "$MESSAGE

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

# Push
echo ""
echo "🚀 Pushing to remote..."
git push

echo ""
echo "✅ Done!"

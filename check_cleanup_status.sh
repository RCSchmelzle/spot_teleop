#!/bin/bash
# Check if git cleanup is done and show new size

cd ~/Projects/teleoperation_spot

if pgrep -f "git filter-branch" > /dev/null; then
    echo "Git filter-branch is still running..."
    ps aux | grep "git filter-branch" | grep -v grep
    echo ""
    echo "Current .git size:"
    du -sh .git
else
    echo "Cleanup appears to be complete!"
    echo ""
    echo "Repository size:"
    du -sh .
    echo ""
    echo ".git directory size:"
    du -sh .git
    echo ""
    echo "Largest remaining files:"
    git ls-files -z | xargs -0 ls -l 2>/dev/null | sort -rn -k5 | head -10 | awk '{print $5/1024/1024 " MB", $9}'
    echo ""
    echo "Ready to commit and push to GitHub!"
    echo "See GITHUB_SETUP.md for instructions."
fi
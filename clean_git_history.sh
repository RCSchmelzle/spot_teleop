#!/bin/bash
# Script to remove large files from git history
# WARNING: This rewrites git history - only run on local repo before pushing!

set -e

cd ~/Projects/teleoperation_spot

echo "Creating backup branch..."
git branch backup-before-cleanup 2>/dev/null || echo "Backup branch already exists"

echo "Removing large files from git history..."
echo "This may take a few minutes..."

# Remove the mcap bag file
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch system_ws/xtion_test_bag/xtion_test_bag_0.mcap' \
  --prune-empty --tag-name-filter cat -- --all

# Remove vocabulary files (will be downloaded separately)
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch \
    "*/Vocabulary/ORBvoc.txt" \
    "*/Vocabulary/ORBvoc.txt.tar.gz" \
    "*/Vocabulary/ORBvoc.txt.tar.gz.1" \
    "*/Vocabulary/ORBvoc.txt.bin"' \
  --prune-empty --tag-name-filter cat -- --all

# Remove large dataset files
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch \
    "*/TUM_IMU/dataset-*.txt" \
    "*/Ground_truth/EuRoC_imu/*.txt"' \
  --prune-empty --tag-name-filter cat -- --all

echo ""
echo "Cleaning up refs and running garbage collection..."
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo ""
echo "Done! Repository cleaned."
echo ""
echo "Checking new size..."
du -sh .git
echo ""
echo "If you want to restore the original, run: git checkout backup-before-cleanup"
echo "To remove backup branch: git branch -D backup-before-cleanup"
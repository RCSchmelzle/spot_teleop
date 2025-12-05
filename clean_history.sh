#!/bin/bash
# Clean git history by removing large files
# Run this to shrink the repository

cd ~/Projects/teleoperation_spot

echo "Backing up current HEAD..."
git tag backup-before-cleanup HEAD

echo ""
echo "Removing large files from ALL commits..."
echo "This will take several minutes..."
echo ""

git filter-branch -f --tree-filter '
  # Remove mcap bags
  find . -name "*.mcap" -type f -delete 2>/dev/null || true

  # Remove large vocabulary files
  find . -path "*/Vocabulary/ORBvoc.txt" -delete 2>/dev/null || true
  find . -path "*/Vocabulary/ORBvoc.txt.tar.gz*" -delete 2>/dev/null || true
  find . -path "*/Vocabulary/ORBvoc.txt.bin" -delete 2>/dev/null || true

  # Remove dataset files
  find . -path "*/TUM_IMU/dataset-*.txt" -delete 2>/dev/null || true
  find . -path "*/Ground_truth/EuRoC_imu/*.txt" -delete 2>/dev/null || true
' --prune-empty HEAD

echo ""
echo "Cleaning up git database..."
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo ""
echo "Done!"
echo ""
du -sh .git
echo ""
echo "To restore original: git reset --hard backup-before-cleanup"
echo "To remove backup: git tag -d backup-before-cleanup"
#!/bin/bash
# Script to remove nested .git directories and make everything part of the main repo

cd ~/Projects/teleoperation_spot

echo "Removing nested .git directories (requires sudo)..."
sudo rm -rf ./_multi_orbslam_ws/src/orb_slam3_ros/.git
sudo rm -rf ./_multi_orbslam_ws/src/ORB_SLAM3/.git

echo "Verifying all nested .git directories are removed..."
find . -name ".git" -type d | grep -v "^\./.git$"

echo ""
echo "Adding all source files to main repo..."
git add -A

echo ""
echo "Checking git status..."
git status

echo ""
echo "Done! You can now commit with:"
echo "  git commit -m 'Include all nested sources directly in main repo'"
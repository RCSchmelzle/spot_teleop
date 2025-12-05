#!/bin/bash
# Fix permissions on large files and clean git history properly

cd ~/Projects/teleoperation_spot

echo "Fixing file permissions (requires sudo)..."
sudo chown -R rysch01:rysch01 _multi_orbslam_ws/src/ORB_SLAM3/
sudo chown -R rysch01:rysch01 _multi_orbslam_ws/src/orb_slam3_ros/

echo ""
echo "Manually removing large files from working directory..."
rm -f _multi_orbslam_ws/src/ORB_SLAM3/Vocabulary/*.tar.gz*
rm -f _multi_orbslam_ws/src/ORB_SLAM3/Vocabulary/ORBvoc.txt
rm -f _multi_orbslam_ws/src/orb_slam3_ros/orb_slam3/Vocabulary/ORBvoc.txt.bin
rm -f _multi_orbslam_ws/src/ORB_SLAM3/Examples*/*/TUM_IMU/dataset-*.txt
rm -f _multi_orbslam_ws/src/ORB_SLAM3/evaluation/Ground_truth/EuRoC_imu/*.txt
rm -f _multi_orbslam_ws/src/orb_slam3_ros/evaluation/Ground_truth/EuRoC_imu/*.txt

rm -f cpp/ORB_SLAM3/Vocabulary/*.tar.gz*
rm -f cpp/ORB_SLAM3/Vocabulary/ORBvoc.txt
rm -f cpp/ORB_SLAM3/Examples*/*/TUM_IMU/dataset-*.txt
rm -f cpp/ORB_SLAM3/evaluation/Ground_truth/EuRoC_imu/*.txt

echo ""
echo "Committing deletion of large files..."
git add -A
git commit -m "Remove large vocabulary and dataset files (see LARGE_FILES.md)"

echo ""
echo "Now cleaning git history..."
git tag -d backup-before-cleanup 2>/dev/null || true

export FILTER_BRANCH_SQUELCH_WARNING=1

git filter-branch -f --tree-filter '
  find . -name "*.mcap" -type f -delete 2>/dev/null || true
  find . -path "*/Vocabulary/ORBvoc.txt" -delete 2>/dev/null || true
  find . -path "*/Vocabulary/ORBvoc.txt.tar.gz*" -delete 2>/dev/null || true
  find . -path "*/Vocabulary/ORBvoc.txt.bin" -delete 2>/dev/null || true
  find . -path "*/TUM_IMU/dataset-*.txt" -delete 2>/dev/null || true
  find . -path "*/Ground_truth/EuRoC_imu/*.txt" -delete 2>/dev/null || true
' --prune-empty HEAD

rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo ""
echo "=========================================="
echo "Cleanup complete!"
echo "=========================================="
du -sh .git
echo ""
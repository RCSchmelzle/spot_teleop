# Large Files Not Included in Repository

Due to size constraints, the following large files are excluded from the git repository.
Download them separately as needed:

## ORB-SLAM3 Vocabulary Files

Download ORBvoc.txt (139 MB) from the ORB-SLAM2 repository:
```bash
cd ~/Projects/teleoperation_spot/cpp/ORB_SLAM3/Vocabulary
wget https://github.com/raulmur/ORB_SLAM2/raw/master/Vocabulary/ORBvoc.txt
```

Alternative compressed versions may already exist in:
- `cpp/ORB_SLAM3/Vocabulary/ORBvoc.txt.tar.gz`
- Extract with: `tar -xzf ORBvoc.txt.tar.gz`

## TUM IMU Dataset Files (for testing)

These are example IMU datasets from TUM, used for testing SLAM algorithms.
They are located in `cpp/ORB_SLAM3/Examples/*/TUM_IMU/dataset-*.txt`

Download from: https://vision.in.tum.de/data/datasets/visual-inertial-dataset

Or regenerate using ORB-SLAM3's example scripts if needed.

## EuRoC Ground Truth Files

Located in `cpp/ORB_SLAM3/evaluation/Ground_truth/EuRoC_imu/*.txt`

Download from: https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

## Binary Vocabulary File

The binary vocabulary file `ORBvoc.txt.bin` can be regenerated from `ORBvoc.txt`
using ORB-SLAM3's vocabulary conversion tools if needed.

## Notes

- These files are only needed for running specific examples and tests
- For basic RGB-D SLAM with Xtion cameras, only `ORBvoc.txt` is required
- Camera calibration and configuration files are included in the repository
#!/bin/bash
# Extrinsic Camera Calibration Tool
#
# This script calibrates rigid camera extrinsics using ORB-SLAM3 trajectories
# and multi-trajectory alignment (time alignment + hand-eye calibration).
#
# Features:
#   1. Live capture: Bring up cameras, record bag, then calibrate
#   2. Existing bags: Process previously recorded bags
#   3. Full pipeline: ORB-SLAM3 → Trajectory → Time sync → Hand-eye → Extrinsics

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Paths
DATASETS_DIR="$HOME/Projects/teleoperation_spot/datasets"
BAGS_DIR="$DATASETS_DIR/xtion_calibration_test/bags"
VOCAB_PATH="$HOME/Projects/teleoperation_spot/cpp/ORB_SLAM3/Vocabulary/ORBvoc.txt"
XTION_CONFIG="$SCRIPT_DIR/src/xtion_bringup/config/xtion_mapping.yaml"

# Ensure bags directory exists
mkdir -p "$BAGS_DIR"

echo -e "${CYAN}======================================================${NC}"
echo -e "${CYAN}   Multi-Camera Extrinsic Calibration Tool${NC}"
echo -e "${CYAN}   ORB-SLAM3 + Hand-Eye Calibration${NC}"
echo -e "${CYAN}======================================================${NC}"
echo

# Function to check if xtion_bringup package is available
check_xtion_bringup() {
    if [ ! -f "$XTION_CONFIG" ]; then
        echo -e "${YELLOW}Warning: Xtion mapping config not found at $XTION_CONFIG${NC}"
        echo -e "${YELLOW}Live recording option will not be available${NC}"
        return 1
    fi
    return 0
}

# Function to record new calibration bag
record_new_bag() {
    echo
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Live Camera Recording${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo

    # Source ROS workspace
    source /opt/ros/jazzy/setup.bash
    source install/setup.bash

    # Create timestamped bag name
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local bag_name="xtion_calib_${timestamp}.bag"
    local bag_path="$BAGS_DIR/$timestamp/$bag_name"
    mkdir -p "$BAGS_DIR/$timestamp"

    echo -e "${YELLOW}Starting Xtion cameras...${NC}"
    echo -e "${YELLOW}This will launch all cameras defined in xtion_mapping.yaml${NC}"
    echo

    # Launch xtion cameras in background
    ros2 launch xtion_bringup multi_xtion.launch.py > /dev/null 2>&1 &
    XTION_PID=$!
    sleep 5  # Wait for cameras to initialize

    # Check if launch succeeded
    if ! ps -p $XTION_PID > /dev/null; then
        echo -e "${RED}Failed to launch Xtion cameras${NC}"
        return 1
    fi

    echo -e "${GREEN}Cameras launched successfully!${NC}"
    echo

    # Get list of topics to record
    echo -e "${YELLOW}Detecting camera topics...${NC}"
    local topics=$(ros2 topic list | grep -E "/(cam[^/]+)/(rgb/image_raw|depth_raw/image|rgb/camera_info)" | tr '\n' ' ')

    if [ -z "$topics" ]; then
        echo -e "${RED}No camera topics found!${NC}"
        kill $XTION_PID 2>/dev/null
        return 1
    fi

    echo -e "${GREEN}Topics to record:${NC}"
    for topic in $topics; do
        echo -e "  - $topic"
    done
    echo

    # Recording instructions
    echo -e "${CYAN}======================================================${NC}"
    echo -e "${CYAN}RECORDING INSTRUCTIONS${NC}"
    echo -e "${CYAN}======================================================${NC}"
    echo
    echo -e "${YELLOW}For good calibration:${NC}"
    echo "  1. Move the camera rig around smoothly"
    echo "  2. Include rotations (not just translations)"
    echo "  3. Point at textured surfaces (posters, patterns)"
    echo "  4. Avoid blank walls and low-light areas"
    echo "  5. Record for 30-60 seconds"
    echo "  6. Move SLOWLY to avoid motion blur"
    echo
    echo -e "${GREEN}Press ENTER to start recording...${NC}"
    read

    # Start recording
    echo -e "${YELLOW}Recording to: $bag_path${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop recording${NC}"
    echo

    # Record bag
    ros2 bag record -o "$bag_path" $topics &
    RECORD_PID=$!

    # Wait for user to stop (Ctrl+C)
    trap "echo ''; echo -e '${YELLOW}Stopping recording...${NC}'; kill $RECORD_PID 2>/dev/null; kill $XTION_PID 2>/dev/null; trap - INT" INT
    wait $RECORD_PID 2>/dev/null || true
    trap - INT

    # Stop cameras
    kill $XTION_PID 2>/dev/null || true
    sleep 2

    echo
    echo -e "${GREEN}Recording complete!${NC}"
    echo -e "${GREEN}Bag saved to: $bag_path${NC}"
    echo

    # Set this as the selected bag
    selected_bag="$bag_path"
    session_dir="$BAGS_DIR/$timestamp"
}

# Function to list available bags
list_bags() {
    echo -e "${GREEN}Available bag files:${NC}"
    local i=1
    bags=()

    # Find all .bag directories
    while IFS= read -r -d '' bag; do
        bags+=("$bag")
        echo -e "  ${YELLOW}[$i]${NC} $(basename $(dirname $bag))/$(basename $bag)"
        ((i++))
    done < <(find "$BAGS_DIR" -name "*.bag" -type d -print0 2>/dev/null | sort -z)

    # Also check current directory for bags
    while IFS= read -r -d '' bag; do
        bags+=("$bag")
        echo -e "  ${YELLOW}[$i]${NC} $bag (in system_ws)"
        ((i++))
    done < <(find "$SCRIPT_DIR" -maxdepth 1 -name "*.bag" -type d -print0 2>/dev/null | sort -z)
}

# Function to select bag
select_bag() {
    list_bags

    local num_bags=${#bags[@]}

    echo
    if [ $num_bags -eq 0 ]; then
        echo -e "${YELLOW}No existing bags found${NC}"
        if check_xtion_bringup; then
            echo -e "${GREEN}You can record a new bag using option [0]${NC}"
        fi
        echo
        read -p "Select [0] to record new bag, or 'q' to quit: " selection
    else
        if check_xtion_bringup; then
            echo -e "  ${GREEN}[0]${NC} ${CYAN}[NEW] Record new calibration bag with live cameras${NC}"
        fi
        echo
        read -p "Select bag number ([0] for live, [1-$num_bags] for existing, 'q' to quit): " selection
    fi

    if [[ "$selection" == "q" ]]; then
        exit 0
    fi

    # Option 0: Record new bag
    if [[ "$selection" == "0" ]]; then
        if ! check_xtion_bringup; then
            echo -e "${RED}Live recording not available (xtion_bringup not configured)${NC}"
            exit 1
        fi
        record_new_bag
        return
    fi

    # Existing bag selection
    if ! [[ "$selection" =~ ^[0-9]+$ ]] || [ "$selection" -lt 1 ] || [ "$selection" -gt $num_bags ]; then
        echo -e "${RED}Invalid selection${NC}"
        exit 1
    fi

    selected_bag="${bags[$((selection-1))]}"
    echo -e "${GREEN}Selected:${NC} $selected_bag"
}

# Function to organize bag into session folder
organize_bag() {
    local bag_path="$1"
    local bag_name=$(basename "$bag_path")

    # Check if already in session folder structure
    if [[ "$bag_path" == *"/bags/"*"/recording.bag" ]] || [[ "$bag_path" == *"/bags/"*"/"*.bag ]]; then
        # Already organized
        session_dir=$(dirname "$bag_path")
        echo -e "${GREEN}Bag already in session folder:${NC} $session_dir"
        return
    fi

    # Create timestamp-based session folder
    local timestamp=$(date +%Y-%m-%d_%H-%M-%S)
    session_dir="$BAGS_DIR/$timestamp"
    mkdir -p "$session_dir"

    # Move bag to session folder
    echo -e "${YELLOW}Moving bag to session folder...${NC}"
    mv "$bag_path" "$session_dir/"
    selected_bag="$session_dir/$bag_name"

    echo -e "${GREEN}Created session folder:${NC} $session_dir"
}

# Function to check for existing config
check_config() {
    config_dir="$session_dir/orbslam_config"
    traj_dir="$session_dir/trajectories"

    if [ -d "$config_dir" ] && [ -f "$config_dir/camera_mapping.yaml" ]; then
        echo
        echo -e "${YELLOW}Existing configuration found${NC}"
        read -p "Use existing config (y) or reconfigure (n)? [y/n]: " use_existing

        if [[ "$use_existing" == "y" ]]; then
            return 0  # Use existing
        else
            # User wants to reconfigure - delete existing config and trajectories
            echo -e "${YELLOW}Deleting existing configuration and trajectories...${NC}"
            rm -rf "$config_dir"
            rm -rf "$traj_dir"
            echo -e "${GREEN}Cleaned up for reconfiguration${NC}"
            return 1  # Reconfigure
        fi
    else
        return 1  # No config, need to configure
    fi
}

# Function to configure cameras
configure_cameras() {
    echo
    echo -e "${BLUE}Configuring cameras...${NC}"

    # Run GUI to map cameras
    python3 "$SCRIPT_DIR/scripts/bag_camera_mapper_gui.py" "$selected_bag" "$session_dir/orbslam_config"

    if [ $? -ne 0 ]; then
        echo -e "${RED}Camera configuration failed${NC}"
        exit 1
    fi

    # Generate ORB-SLAM3 configs
    echo -e "${YELLOW}Generating ORB-SLAM3 configs...${NC}"
    python3 "$SCRIPT_DIR/scripts/generate_orbslam_configs.py" "$session_dir/orbslam_config"

    echo -e "${GREEN}Configuration complete${NC}"
}

# Function to run ORB-SLAM3
run_orbslam() {
    local camera_name="$1"
    local config_file="$session_dir/orbslam_config/${camera_name}_rgbd.yaml"
    local traj_dir="$session_dir/trajectories"

    mkdir -p "$traj_dir"

    echo
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Running ORB-SLAM3 on $camera_name${NC}"
    echo -e "${BLUE}========================================${NC}"

    # Source ROS and workspace
    source /opt/ros/jazzy/setup.bash
    source install/setup.bash
    export LD_LIBRARY_PATH="$HOME/Projects/teleoperation_spot/cpp/ORB_SLAM3/lib:$LD_LIBRARY_PATH"

    # Get topic remapping from mapping file
    local mapping_file="$session_dir/orbslam_config/camera_mapping.yaml"
    local rgb_topic=$(python3 -c "
import yaml
with open('$mapping_file', 'r') as f:
    data = yaml.safe_load(f)
    for cam in data['cameras']:
        if cam['name'] == '$camera_name':
            print(cam['topics']['rgb'])
            break
")
    local depth_topic=$(python3 -c "
import yaml
with open('$mapping_file', 'r') as f:
    data = yaml.safe_load(f)
    for cam in data['cameras']:
        if cam['name'] == '$camera_name':
            print(cam['topics']['depth'])
            break
")

    echo -e "${YELLOW}Topics:${NC}"
    echo "  RGB: $rgb_topic"
    echo "  Depth: $depth_topic"
    echo
    echo -e "${YELLOW}Starting bag playback in background...${NC}"

    # Play bag in background (no loop - play once)
    ros2 bag play "$selected_bag" &
    BAG_PID=$!
    sleep 2  # Wait for bag to start

    echo -e "${GREEN}Running ORB-SLAM3... (Ctrl+C to stop early)${NC}"
    echo

    # Setup cleanup trap
    cleanup_orbslam() {
        echo
        echo -e "${YELLOW}Stopping ORB-SLAM3...${NC}"
        kill $ORBSLAM_PID 2>/dev/null || true
        kill $BAG_PID 2>/dev/null || true
        wait $ORBSLAM_PID 2>/dev/null || true
        sleep 1
    }
    trap cleanup_orbslam INT

    # Run ORB-SLAM3
    cd "$traj_dir"
    "$SCRIPT_DIR/install/orbslam3/lib/orbslam3/rgbd" \
        "$VOCAB_PATH" \
        "$config_file" \
        --ros-args \
        -r /camera/rgb:="$rgb_topic" \
        -r /camera/depth:="$depth_topic" &
    ORBSLAM_PID=$!

    # Monitor bag playback - when it finishes, stop ORB-SLAM3
    wait $BAG_PID 2>/dev/null || true
    echo
    echo -e "${YELLOW}Bag playback finished, stopping ORB-SLAM3...${NC}"
    sleep 2  # Give ORB-SLAM3 a moment to finish processing
    kill $ORBSLAM_PID 2>/dev/null || true
    wait $ORBSLAM_PID 2>/dev/null || true

    # Cleanup
    trap - INT
    sleep 1

    # Rename trajectory with camera name
    if [ -f "KeyFrameTrajectory.txt" ]; then
        mv "KeyFrameTrajectory.txt" "${camera_name}_KeyFrameTrajectory.txt"
        echo
        echo -e "${GREEN}Trajectory saved:${NC} $traj_dir/${camera_name}_KeyFrameTrajectory.txt"
    else
        echo
        echo -e "${YELLOW}No trajectory file found (expected at KeyFrameTrajectory.txt)${NC}"
    fi

    cd "$SCRIPT_DIR"
}

# Main workflow
main() {
    while true; do
        # Step 1: Select bag (existing or record new)
        select_bag

        # Step 2: Organize into session folder (if not already from live recording)
        if [ ! -d "$session_dir" ]; then
            organize_bag "$selected_bag"
        fi

        # Step 3: Check/create configuration
        if ! check_config; then
            configure_cameras
        fi

        # Step 4-6: Process all cameras sequentially
        local mapping_file="$session_dir/orbslam_config/camera_mapping.yaml"
        cameras=($(python3 -c "
import yaml
with open('$mapping_file', 'r') as f:
    data = yaml.safe_load(f)
    for cam in data['cameras']:
        print(cam['name'])
"))

        echo
        echo -e "${GREEN}Found ${#cameras[@]} cameras to process${NC}"
        for cam in "${cameras[@]}"; do
            echo -e "  - $cam"
        done
        echo

        # Process all cameras sequentially (bag playback uses ROS timestamps)
        for camera_name in "${cameras[@]}"; do
            echo
            echo -e "${BLUE}========================================${NC}"
            echo -e "${BLUE}Processing $camera_name${NC}"
            echo -e "${BLUE}========================================${NC}"

            run_orbslam "$camera_name"

            echo
            if [ -f "$session_dir/trajectories/${camera_name}_KeyFrameTrajectory.txt" ]; then
                echo -e "${GREEN}✓${NC} Trajectory saved: ${camera_name}_KeyFrameTrajectory.txt"
            else
                echo -e "${YELLOW}⚠${NC} No trajectory found for $camera_name"
            fi
        done

        echo
        echo -e "${GREEN}All cameras processed!${NC}"

        # Step 7: Calibrate camera extrinsics (optional)
        if [ ${#cameras[@]} -gt 1 ]; then
            echo
            echo -e "${BLUE}========================================${NC}"
            echo -e "${BLUE}Camera Extrinsic Calibration${NC}"
            echo -e "${BLUE}========================================${NC}"
            echo
            read -p "Calibrate camera extrinsics from trajectories? [y/n]: " do_calibration

            if [[ "$do_calibration" == "y" ]]; then
                echo -e "${YELLOW}Running extrinsic calibration...${NC}"
                echo

                python3 "$SCRIPT_DIR/scripts/calibrate_cameras_from_trajectories.py" "$session_dir"

                if [ $? -eq 0 ]; then
                    echo
                    echo -e "${GREEN}Calibration complete!${NC}"
                    echo "Extrinsics saved to: $session_dir/orbslam_config/extrinsics/"
                else
                    echo
                    echo -e "${RED}Calibration failed${NC}"
                fi
            fi
        fi

        echo
        read -p "Process another bag? [y/n]: " another_bag
        if [[ "$another_bag" != "y" ]]; then
            break
        fi
        echo
    done

    echo
    echo -e "${GREEN}Done!${NC}"
}

main
#!/bin/bash
# Interactive ORB-SLAM3 Bag Processor
# Handles bag organization, camera configuration, and trajectory generation

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Parse command line arguments
HEADLESS=false
BAG_NUMBER=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --headless|-H)
            HEADLESS=true
            shift
            ;;
        --bag|-b)
            BAG_NUMBER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--headless|-H] [--bag|-b <number>]"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Paths
DATASETS_DIR="$HOME/Projects/teleoperation_spot/datasets"
BAGS_DIR="$DATASETS_DIR/xtion_calibration_test/bags"
VOCAB_PATH="$HOME/Projects/teleoperation_spot/cpp/ORB_SLAM3/Vocabulary/ORBvoc.txt"

# Ensure bags directory exists
mkdir -p "$BAGS_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}ORB-SLAM3 Bag Processor${NC}"
echo -e "${BLUE}========================================${NC}"
echo

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

    if [ ${#bags[@]} -eq 0 ]; then
        echo -e "${RED}No bag files found!${NC}"
        echo "Place .bag directories in $BAGS_DIR"
        exit 1
    fi
}

# Function to select bag
select_bag() {
    list_bags
    echo

    if [[ -n "$BAG_NUMBER" ]]; then
        # Use provided bag number
        selection="$BAG_NUMBER"
        echo -e "${GREEN}Using bag number from command line:${NC} $selection"
    else
        read -p "Select bag number (or 'q' to quit): " selection

        if [[ "$selection" == "q" ]]; then
            exit 0
        fi
    fi

    if ! [[ "$selection" =~ ^[0-9]+$ ]] || [ "$selection" -lt 1 ] || [ "$selection" -gt ${#bags[@]} ]; then
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

        if [[ "$HEADLESS" == "true" ]]; then
            # In headless mode, use existing config
            echo -e "${GREEN}Using existing configuration (headless mode)${NC}"
            return 0
        fi

        read -p "Reconfigure? [Y/n]: " reconfigure
        reconfigure=${reconfigure:-y}  # Default to 'y' if empty

        if [[ "$reconfigure" == "n" ]] || [[ "$reconfigure" == "N" ]]; then
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

    # Run camera mapper (GUI or headless)
    if [[ "$HEADLESS" == "true" ]]; then
        python3 "$SCRIPT_DIR/scripts/bag_camera_mapper_gui.py" --headless "$selected_bag" "$session_dir/orbslam_config"
    else
        python3 "$SCRIPT_DIR/scripts/bag_camera_mapper_gui.py" "$selected_bag" "$session_dir/orbslam_config"
    fi

    if [ $? -ne 0 ]; then
        echo -e "${RED}Camera configuration failed${NC}"
        exit 1
    fi

    # Generate ORB-SLAM3 configs
    echo -e "${YELLOW}Generating ORB-SLAM3 configs...${NC}"
    python3 "$SCRIPT_DIR/scripts/generate_orbslam_configs.py" "$session_dir/orbslam_config"

    echo -e "${GREEN}Configuration complete${NC}"
}

# Function to select camera for ORB-SLAM3
select_camera() {
    echo
    echo -e "${GREEN}Available cameras:${NC}"

    local mapping_file="$session_dir/orbslam_config/camera_mapping.yaml"
    cameras=($(python3 -c "
import yaml
with open('$mapping_file', 'r') as f:
    data = yaml.safe_load(f)
    for cam in data['cameras']:
        print(cam['name'])
"))

    for i in "${!cameras[@]}"; do
        echo -e "  ${YELLOW}[$((i+1))]${NC} ${cameras[$i]}"
    done

    echo
    read -p "Select camera number: " cam_selection

    if ! [[ "$cam_selection" =~ ^[0-9]+$ ]] || [ "$cam_selection" -lt 1 ] || [ "$cam_selection" -gt ${#cameras[@]} ]; then
        echo -e "${RED}Invalid selection${NC}"
        return 1
    fi

    selected_camera="${cameras[$((cam_selection-1))]}"
    echo -e "${GREEN}Selected camera:${NC} $selected_camera"
    return 0
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
        # Step 1: Select bag
        select_bag

        # Step 2: Organize into session folder
        organize_bag "$selected_bag"

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

        # Skip initial SLAM run - let multi-SLAM calibration handle all runs
        # (Commented out to avoid duplicate SLAM execution)
        # # Process all cameras sequentially (bag playback uses ROS timestamps)
        # for camera_name in "${cameras[@]}"; do
        #     echo
        #     echo -e "${BLUE}========================================${NC}"
        #     echo -e "${BLUE}Processing $camera_name${NC}"
        #     echo -e "${BLUE}========================================${NC}"
        #
        #     run_orbslam "$camera_name"
        #
        #     echo
        #     if [ -f "$session_dir/trajectories/${camera_name}_KeyFrameTrajectory.txt" ]; then
        #         echo -e "${GREEN}✓${NC} Trajectory saved: ${camera_name}_KeyFrameTrajectory.txt"
        #     else
        #         echo -e "${YELLOW}⚠${NC} No trajectory found for $camera_name"
        #     fi
        # done
        #
        # echo
        # echo -e "${GREEN}All cameras processed!${NC}"

        # Step 7: Stochastic multi-SLAM calibration (automatic)
        if [ ${#cameras[@]} -gt 1 ]; then
            echo
            echo -e "${BLUE}========================================${NC}"
            echo -e "${BLUE}Stochastic Multi-SLAM Calibration${NC}"
            echo -e "${BLUE}========================================${NC}"
            echo

            # Check if atlas/trajectories exist and ask about reset
            atlas_dir="$session_dir/atlas"
            traj_dir="$session_dir/trajectories"
            if [[ "$HEADLESS" == "true" ]]; then
                # In headless mode, always reset
                echo -e "${GREEN}Headless mode: resetting atlas and trajectories${NC}"
                rm -rf "$atlas_dir"/* "$traj_dir"/* 2>/dev/null
            else
                if [ -d "$atlas_dir" ] && [ "$(ls -A "$atlas_dir" 2>/dev/null)" ]; then
                    atlas_size=$(du -sh "$atlas_dir" 2>/dev/null | cut -f1)
                    echo -e "${YELLOW}Existing atlas found ($atlas_size)${NC}"
                    read -p "Reset atlas and trajectories? [Y/n]: " reset_atlas
                    reset_atlas=${reset_atlas:-Y}
                    if [[ "$reset_atlas" =~ ^[Yy] ]]; then
                        echo "Clearing atlas and trajectories..."
                        rm -rf "$atlas_dir"/* "$traj_dir"/* 2>/dev/null
                    else
                        echo "Keeping existing data (will merge new runs)"
                    fi
                    echo
                fi
            fi

            echo -e "${YELLOW}Running multi-SLAM stochastic calibration...${NC}"
            echo -e "${YELLOW}This will take several minutes...${NC}"
            echo

            if [[ "$HEADLESS" == "true" ]]; then
                # Pass --headless to Python script (uses defaults)
                python3 "$SCRIPT_DIR/scripts/multi_slam_calibration.py" "$session_dir" --headless
            else
                # Interactive mode - Python script will prompt for settings
                python3 "$SCRIPT_DIR/scripts/multi_slam_calibration.py" "$session_dir"
            fi

            if [ $? -eq 0 ]; then
                echo
                echo -e "${GREEN}Stochastic calibration complete!${NC}"
                echo "Results saved to: $session_dir/orbslam_config/stochastic_extrinsics/"
            else
                echo
                echo -e "${RED}Calibration failed${NC}"
            fi
        fi

        echo
        if [[ "$HEADLESS" == "true" ]]; then
            # In headless mode, exit after one bag
            break
        fi
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
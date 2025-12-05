from pathlib import Path
import yaml

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def load_cameras_config():
    share_dir = Path(get_package_share_directory('xtion_bringup'))
    yaml_path = share_dir / 'config' / 'xtion_cameras.yaml'

    if not yaml_path.exists():
        raise RuntimeError(
            f"[multi_xtion] Config file not found: {yaml_path}\n"
            f"Run: python3 -m xtion_bringup.gen_xtion_cameras"
        )

    with yaml_path.open('r') as f:
        data = yaml.safe_load(f) or {}

    cameras = data.get('cameras', [])
    if not cameras:
        print("[multi_xtion] WARNING: No cameras in config file.")
    return cameras


def generate_launch_description():
    cameras = load_cameras_config()
    actions = []

    for cam in cameras:
        name = cam.get('name', 'camera')
        namespace = cam.get('namespace', '')
        device_id = cam.get('device_id', '')

        if not device_id:
            raise RuntimeError(f"[multi_xtion] Camera entry missing device_id: {cam}")

        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare('asus_xtion'),
                        'launch',
                        'single_asus_xtion.launch.py',
                    ])
                ),
                launch_arguments={
                    'camera_name': name,
                    'camera_namespace': namespace,
                    'device_id': device_id,
                }.items(),
            )
        )

    return LaunchDescription(actions)

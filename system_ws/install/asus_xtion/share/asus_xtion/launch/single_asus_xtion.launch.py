from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():
    camera_name = LaunchConfiguration('camera_name')
    camera_namespace = LaunchConfiguration('camera_namespace')
    device_id = LaunchConfiguration('device_id')

    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_name',
            default_value='camera',
            description='Name of the camera node'
        ),
        DeclareLaunchArgument(
            'camera_namespace',
            default_value='',
            description='Namespace for this camera (e.g. cam0, cam1)'
        ),
        DeclareLaunchArgument(
            'device_id',
            default_value='',
            description='OpenNI2 device ID / URI (e.g. 1d27/0601@1/9)'
        ),

        Node(
            package='openni2_camera',
            executable='openni2_camera_driver', 
            namespace=camera_namespace,
            name=camera_name,
            parameters=[{
                'device_id': device_id,
            }],
            output='screen',
        ),
    ])

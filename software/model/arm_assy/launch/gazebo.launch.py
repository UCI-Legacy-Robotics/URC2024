import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

def generate_launch_description():
    pkg_arm_assy = get_package_share_directory('arm_assy')
    default_urdf_path = PathJoinSubstitution([
        pkg_arm_assy,
        'urdf',
        'Arm_Final_PLease.urdf'
    ])
    urdf_file_arg = DeclareLaunchArgument(
        name='urdf_file',
        default_value=default_urdf_path,
        description='Absolute path to the robot URDF file'
    )

    start_gz_sim = ExecuteProcess(
        cmd=[
            'gz', 'sim',
            '-r',
            'empty.sdf'
        ],
        output='screen'
    )

    spawn_robot_cmd = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-file', LaunchConfiguration('urdf_file'),
            '-name', 'arm_assy',
            '-x', '0',
            '-y', '0',
            '-z', '0.5'
        ],
        output='screen'
    )

    ld = LaunchDescription()
    ld.add_action(urdf_file_arg)
    ld.add_action(start_gz_sim)
    ld.add_action(spawn_robot_cmd)
    return ld

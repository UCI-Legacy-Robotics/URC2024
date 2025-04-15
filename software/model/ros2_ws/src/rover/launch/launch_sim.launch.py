import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg = 'rover'

    # 1) Robot State Publisher (rsp.launch.py)
    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(pkg), 'launch', 'rsp.launch.py')
        ),
        launch_arguments={'use_sim_time': 'true'}.items()
    )

    # 2) Gazebo (empty world)
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'),
                         'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': '-r empty.sdf'}.items()
    )

    # 3) Spawn the robot after 5s
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-world', 'empty',
            '-topic', 'robot_description',  # match your actual topic
            '-entity', 'rover',
            '-allow_renaming', 'true'
        ]
    )
    spawn_timer = TimerAction(
        period=5.0,
        actions=[spawn_entity]
    )

    # 4) Load joint_state_broadcaster at 7s
    load_jsb = ExecuteProcess(
        cmd=[
            'ros2', 'control', 'load_controller', '--set-state', 'active',
            'joint_state_broadcaster'
        ],
        output='screen'
    )
    jsb_timer = TimerAction(
        period=7.0,
        actions=[load_jsb]
    )

    # 5) Load joint_trajectory_controller at 9s
    load_jtc = ExecuteProcess(
        cmd=[
            'ros2', 'control', 'load_controller', '--set-state', 'active',
            'joint_trajectory_controller'
        ],
        output='screen'
    )
    jtc_timer = TimerAction(
        period=9.0,
        actions=[load_jtc]
    )

    return LaunchDescription([
        rsp,
        gazebo,
        spawn_timer,
        jsb_timer,
        jtc_timer,
    ])

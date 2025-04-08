import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

def generate_launch_description():
    pkg_arm_assy = get_package_share_directory('rover')
    default_urdf_path = PathJoinSubstitution([
        pkg_arm_assy,
        'urdf',
        'rover.urdf'
    ])
    
    # Path to the controllers config file
    controllers_config_path = PathJoinSubstitution([
        pkg_arm_assy,
        'config',
        'controllers.yaml'
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
            '-name', 'rover',
            '-x', '0',
            '-y', '0',
            '-z', '0.5'
        ],
        output='screen'
    )
    
    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': LaunchConfiguration('urdf_file')}]
    )
    
    # Load controllers from YAML config
    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[controllers_config_path],
        output='screen'
    )
    
    # Spawn controllers
    # Modify the list to include all controllers you need from your controllers.yaml
    controller_names = ['arm_controller', 'joint_state_broadcaster']  # Update with your actual controller names
    
    spawner_nodes = []
    for controller in controller_names:
        spawner_nodes.append(
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=[controller],
                output='screen',
            )
        )
    
    # Create event handlers to spawn controllers after the robot is launched
    delay_controller_spawners = []
    for spawner in spawner_nodes:
        delay_controller_spawners.append(
            RegisterEventHandler(
                event_handler=OnProcessStart(
                    target_action=spawn_robot_cmd,
                    on_start=[spawner],
                )
            )
        )

    ld = LaunchDescription()
    ld.add_action(urdf_file_arg)
    ld.add_action(start_gz_sim)
    ld.add_action(robot_state_publisher)
    ld.add_action(spawn_robot_cmd)
    ld.add_action(controller_manager)
    
    # Add all controller spawners with delays
    for spawner_event in delay_controller_spawners:
        ld.add_action(spawner_event)
        
    return ld
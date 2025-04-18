import sys
import math
from launch import LaunchDescription
import launch.actions
import launch_ros.actions


platform = 0

for arg in sys.argv:
    if arg.startswith("platform:="):
        platform = float(arg.split(":=")[1])


platform = platform * (math.pi/180)

def generate_launch_description():
    return LaunchDescription([
        launch_ros.actions.Node(
            package='rover',
            executable='control.py',
            output='screen',
            arguments=[str(platform)]),
    ])

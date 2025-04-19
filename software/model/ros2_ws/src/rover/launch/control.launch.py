import sys
import math
from launch import LaunchDescription
import launch.actions
import launch_ros.actions


platform = 0

for arg in sys.argv:
    if arg.startswith("platform:="):
        platform = float(arg.split(":=")[1])
    if arg.startswith("linkage1:="):
        linkage1 = float(arg.split(":=")[1])
    if arg.startswith("linkage3:="):
        linkage3 = float(arg.split(":=")[1])
    if arg.startswith("wrist:="):
        wrist = float(arg.split(":=")[1])
    if arg.startswith("manipulator_wrist:="):
        manipulator_wrist = float(arg.split(":=")[1])
    if arg.startswith("top_claw:="):
        top_claw = float(arg.split(":=")[1])


platform          = platform          * (math.pi/180)
linkage1          = linkage1          * (math.pi/180)
linkage3          = linkage3          * (math.pi/180)
wrist             = wrist             * (math.pi/180)
manipulator_wrist = manipulator_wrist * (math.pi/180)
top_claw          = top_claw          * (math.pi/180)

def generate_launch_description():
    return LaunchDescription([
        launch_ros.actions.Node(
            package='rover',
            executable='control.py',
            output='screen',
            arguments=[str(platform), str(linkage1), str(linkage3), str(wrist), str(manipulator_wrist), str(top_claw)]),
    ])

# ros2 launch rover control.launch.py platform:=100 linkage1:=50 linkage3:=75 wrist:=45 manipulator_wrist:=80 top_claw:=10
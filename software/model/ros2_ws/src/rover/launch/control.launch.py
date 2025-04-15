import sys
import math
from launch import LaunchDescription
import launch.actions
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
import os


ax1 = 0
ax2 = 0
ax3 = 0
ax4 = 0
ax5 = 0
ax6 = 0

for arg in sys.argv:
    if arg.startswith("platform:="):
        ax1 = float(arg.split(":=")[1])
	
    if arg.startswith("linkage1:="):
        ax2 = float(arg.split(":=")[1])

    if arg.startswith("linkage3:="):
        ax3 = float(arg.split(":=")[1])
    
    if arg.startswith("wrist:="):
        ax3 = float(arg.split(":=")[1])

    if arg.startswith("manipulator_wrist:="):
        ax5 = float(arg.split(":=")[1])

    if arg.startswith("top_claw:="):
        ax6 = float(arg.split(":=")[1])

ax1 = ax1 * (math.pi/180)
ax2 = ax2 * (math.pi/180)
ax3 = ax3 * (math.pi/180)
ax4 = ax4 * (math.pi/180)
ax5 = ax5 * (math.pi/180)
ax6 = ax6 * (math.pi/180)

def generate_launch_description():

    pkg_path = os.path.join(get_package_share_directory('rover'))
    controls_path = os.path.join(pkg_path, 'src','control.')


    return LaunchDescription([
        launch_ros.actions.Node(
            package='rover',
            executable='control.py',
            output='screen',
            arguments=[str(ax1), str(ax2), str(ax3), str(ax4), str(ax5), str(ax6)]),
    ])

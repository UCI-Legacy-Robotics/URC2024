source /opt/ros/humble/setup.bash
colcon build --packages-select rover
source install/setup.bash
ros2 launch rover launch_sim.launch.py

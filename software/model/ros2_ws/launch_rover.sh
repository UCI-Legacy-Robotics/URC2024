source /opt/ros/humble/setup.bash
colcon build --packages-select rover
source install/setup.bash
LIBGL_ALWAYS_SOFTWARE=1 ros2 launch rover launch_sim.launch.py

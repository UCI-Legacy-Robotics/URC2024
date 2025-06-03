#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPlanningScene
import time


class MoveItDiagnostics(Node):
    def __init__(self):
        super().__init__('moveit_diagnostics')
        
        # Subscribe to joint states
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Service client for planning scene (to get robot info)
        self.planning_scene_client = self.create_client(
            GetPlanningScene,
            '/get_planning_scene'
        )
        
        self.joint_states_received = False
        
        self.get_logger().info('MoveIt Diagnostics started...')
        
        # Wait for services
        if self.planning_scene_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info('Planning scene service available')
        else:
            self.get_logger().warn('Planning scene service not available')

    def joint_state_callback(self, msg):
        """Print current joint states"""
        if not self.joint_states_received:
            self.get_logger().info('=== CURRENT JOINT STATES ===')
            self.get_logger().info(f'Joint names: {msg.name}')
            self.get_logger().info(f'Joint positions: {[round(pos, 3) for pos in msg.position]}')
            self.get_logger().info(f'Number of joints: {len(msg.name)}')
            self.joint_states_received = True

    def get_planning_scene_info(self):
        """Get information about the robot from planning scene"""
        if not self.planning_scene_client.service_is_ready():
            self.get_logger().error('Planning scene service not ready')
            return
        
        request = GetPlanningScene.Request()
        request.components.components = request.components.ROBOT_STATE
        
        try:
            future = self.planning_scene_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            
            if future.result() is not None:
                response = future.result()
                robot_state = response.scene.robot_state
                
                self.get_logger().info('=== ROBOT STATE FROM PLANNING SCENE ===')
                if robot_state.joint_state.name:
                    self.get_logger().info(f'Joint names: {robot_state.joint_state.name}')
                    self.get_logger().info(f'Joint positions: {[round(pos, 3) for pos in robot_state.joint_state.position]}')
                else:
                    self.get_logger().warn('No joint state in planning scene')
                    
        except Exception as e:
            self.get_logger().error(f'Failed to get planning scene: {e}')

    def print_ros_info(self):
        """Print ROS topic and service information"""
        self.get_logger().info('=== CHECKING ROS TOPICS AND SERVICES ===')
        
        # Check if joint trajectory controller is running
        try:
            import subprocess
            result = subprocess.run(['ros2', 'action', 'list'], 
                                  capture_output=True, text=True, timeout=5)
            actions = result.stdout.strip().split('\n')
            
            trajectory_actions = [action for action in actions if 'trajectory' in action.lower()]
            self.get_logger().info(f'Available trajectory actions: {trajectory_actions}')
            
        except Exception as e:
            self.get_logger().warn(f'Could not check actions: {e}')


def main():
    rclpy.init()
    
    diagnostics = MoveItDiagnostics()
    
    try:
        # Let it run for a few seconds to collect information
        diagnostics.get_logger().info('Collecting diagnostic information...')
        
        # Spin for a bit to get joint states
        for i in range(10):
            rclpy.spin_once(diagnostics, timeout_sec=0.5)
            time.sleep(0.1)
        
        # Get planning scene info
        diagnostics.get_planning_scene_info()
        
        # Print ROS info
        diagnostics.print_ros_info()
        
        diagnostics.get_logger().info('=== DIAGNOSTIC SUMMARY ===')
        diagnostics.get_logger().info('Please check the joint names above and update your arm controller accordingly.')
        diagnostics.get_logger().info('Common joint naming patterns:')
        diagnostics.get_logger().info('- joint_1, joint_2, joint_3, etc.')
        diagnostics.get_logger().info('- shoulder_pan_joint, shoulder_lift_joint, etc.')
        diagnostics.get_logger().info('- base_joint, link1_joint, etc.')
        
    except KeyboardInterrupt:
        diagnostics.get_logger().info('Diagnostics interrupted by user')
    finally:
        diagnostics.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
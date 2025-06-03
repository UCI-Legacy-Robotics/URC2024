#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetMotionPlan, GetCartesianPath
from moveit_msgs.msg import MotionPlanRequest, Constraints, JointConstraint
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.executors import MultiThreadedExecutor
import time

class RobotController(Node):
    def __init__(self):
        super().__init__('arm_controller')
        
        # Define your arm joint names
        self.arm_joint_names = [
            "platform",  # Base rotation
            "linkage1",           # Shoulder
            "linkage3",           # Elbow  
            "wrist",           # Wrist pitch
            "manipulator_wrist"   # Wrist roll
        ]
        
        # Gripper joint
        self.gripper_joint_names = ["top_claw"]
        
        # All joints combined
        self.joint_names = self.arm_joint_names + self.gripper_joint_names
        
        # Service clients for MoveIt2
        self.plan_service = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        self.cartesian_service = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        
        # Publisher for joint trajectory
        self.trajectory_pub = self.create_publisher(
            JointTrajectory, 
            '/joint_trajectory_controller/joint_trajectory', 
            10
        )
        
        # Subscriber for current joint states
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.current_joint_state = None
        
        # Wait for services
        self.get_logger().info("Waiting for services...")
        self.plan_service.wait_for_service()
        self.cartesian_service.wait_for_service()
        self.get_logger().info("Services available!")

    def joint_state_callback(self, msg):
        """Store current joint state"""
        self.current_joint_state = msg

    def plan_to_joint_goal(self, joint_values, group_name="arm", joint_names=None):
        """Plan motion to joint goal using MoveIt2 services"""
        if joint_names is None:
            joint_names = self.joint_names
            
        if len(joint_values) != len(joint_names):
            self.get_logger().error(f"Expected {len(joint_names)} joint values, got {len(joint_values)}")
            return None
            
        # Create motion plan request
        request = GetMotionPlan.Request()
        request.motion_plan_request.group_name = group_name
        request.motion_plan_request.num_planning_attempts = 10
        request.motion_plan_request.allowed_planning_time = 5.0
        
        # Set goal constraints
        goal_constraints = Constraints()
        for i, (joint_name, value) in enumerate(zip(joint_names, joint_values)):
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = joint_name
            joint_constraint.position = value
            joint_constraint.tolerance_above = 0.01
            joint_constraint.tolerance_below = 0.01
            joint_constraint.weight = 1.0
            goal_constraints.joint_constraints.append(joint_constraint)
        
        request.motion_plan_request.goal_constraints = [goal_constraints]
        
        # Call planning service
        future = self.plan_service.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            response = future.result()
            if response.motion_plan_response.error_code.val == 1:  # SUCCESS
                self.get_logger().info("Planning successful!")
                return response.motion_plan_response.trajectory
            else:
                self.get_logger().error(f"Planning failed with error code: {response.motion_plan_response.error_code.val}")
        else:
            self.get_logger().error("Service call failed")
        
        return None

    def execute_trajectory(self, trajectory):
        """Execute the planned trajectory"""
        if trajectory is None:
            self.get_logger().error("No trajectory to execute")
            return False
            
        # Convert RobotTrajectory to JointTrajectory
        joint_traj = JointTrajectory()
        joint_traj.header.stamp = self.get_clock().now().to_msg()
        joint_traj.joint_names = self.joint_names
        
        # Copy trajectory points
        for point in trajectory.joint_trajectory.points:
            traj_point = JointTrajectoryPoint()
            traj_point.positions = point.positions
            traj_point.velocities = point.velocities if point.velocities else [0.0] * len(self.joint_names)
            traj_point.time_from_start = point.time_from_start
            joint_traj.points.append(traj_point)
        
        # Publish trajectory
        self.trajectory_pub.publish(joint_traj)
        self.get_logger().info("Trajectory sent for execution")
        return True

    def move_arm_to_joint_positions(self, arm_joint_values, group_name="arm"):
        """Move only the arm joints (excluding gripper)"""
        if len(arm_joint_values) != len(self.arm_joint_names):
            self.get_logger().error(f"Expected {len(self.arm_joint_names)} arm joint values, got {len(arm_joint_values)}")
            return False
            
        self.get_logger().info(f"Moving arm to joint positions: {dict(zip(self.arm_joint_names, arm_joint_values))}")
        
        trajectory = self.plan_to_joint_goal(arm_joint_values, group_name, joint_names=self.arm_joint_names)
        if trajectory:
            return self.execute_trajectory(trajectory)
        return False

    def move_gripper(self, gripper_position):
        """Control gripper (top_claw) position"""
        self.get_logger().info(f"Moving gripper to position: {gripper_position}")
        
        trajectory = self.plan_to_joint_goal([gripper_position], "gripper", joint_names=self.gripper_joint_names)
        if trajectory:
            return self.execute_trajectory(trajectory)
        return False

    def move_to_joint_positions(self, joint_values, group_name="arm"):
        """Plan and execute motion to joint positions (all joints)"""
        self.get_logger().info(f"Moving to joint positions: {joint_values}")
        
        trajectory = self.plan_to_joint_goal(joint_values, group_name)
        if trajectory:
            return self.execute_trajectory(trajectory)
        return False

    def get_current_joint_positions(self):
        """Get current joint positions"""
        if self.current_joint_state is None:
            self.get_logger().warn("No joint state received yet")
            return None
            
        # Map joint names to positions
        joint_positions = {}
        for name, position in zip(self.current_joint_state.name, self.current_joint_state.position):
            if name in self.joint_names:
                joint_positions[name] = position
                
        return joint_positions

    def plan_cartesian_path(self, waypoints, group_name="arm"):
        """Plan cartesian path through waypoints"""
        request = GetCartesianPath.Request()
        request.group_name = group_name
        request.waypoints = waypoints
        request.max_step = 0.01  # 1cm steps
        request.jump_threshold = 0.0  # Disable jump threshold
        
        future = self.cartesian_service.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            response = future.result()
            if response.fraction > 0.9:  # At least 90% of path achievable
                self.get_logger().info(f"Cartesian planning successful! Fraction: {response.fraction}")
                return response.solution
            else:
                self.get_logger().warn(f"Cartesian planning partially successful. Fraction: {response.fraction}")
                return response.solution
        else:
            self.get_logger().error("Cartesian planning service call failed")
        
        return None

def main():
    rclpy.init()
    
    controller = RobotController()
    
    # Example usage
    try:
        # Wait a bit for joint state
        time.sleep(2)
        
        # Get current positions
        current_pos = controller.get_current_joint_positions()
        if current_pos:
            controller.get_logger().info(f"Current joint positions: {current_pos}")
        
        # Example joint goals (adjust these values for your robot)
        home_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Adjust as needed
        goal_position = [0.5, -0.5, 0.3, 0.0, 0.0, 0.0]  # Example goal
        
        # Move to home position
        controller.get_logger().info("Moving to home position...")
        success = controller.move_to_joint_positions(home_position)
        
        if success:
            time.sleep(3)  # Wait for execution
            controller.get_logger().info("Moving to goal position...")
            controller.move_to_joint_positions(goal_position)
        
        # Keep node alive
        rclpy.spin(controller)
        
    except KeyboardInterrupt:
        controller.get_logger().info("Shutting down...")
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
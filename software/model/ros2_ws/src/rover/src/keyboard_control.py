#!/usr/bin/env python3

import sys
import math
import threading
import time
import rclpy
from rclpy.duration import Duration
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

# Try to import termios for Unix systems
try:
    import termios
    import tty
    TERMIOS_AVAILABLE = True
except ImportError:
    TERMIOS_AVAILABLE = False

class FastRoverController(Node):
    def __init__(self):
        super().__init__('fast_rover_controller')
        
        # Create action client for joint trajectory
        self.action_client = ActionClient(
            self, 
            FollowJointTrajectory, 
            '/joint_trajectory_controller/follow_joint_trajectory'
        )
        
        # Joint positions in degrees
        self.joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.joint_names = [
            'platform', 'linkage1', 'linkage3', 
            'wrist', 'manipulator_wrist', 'top_claw'
        ]
        
        # Control parameters
        self.step_size = 5.0  # degrees per keypress
        self.movement_time = 0.5  # seconds to complete movement (faster!)
        
        # Joint limits in degrees
        self.limits = [
            (-180, 180),  # platform
            (-90, 90),    # linkage1
            (-90, 90),    # linkage3
            (-90, 90),    # wrist
            (-180, 180),  # manipulator_wrist
            (-45, 45)     # top_claw
        ]
        
        self.running = True
        self.current_goal_handle = None
        
        # Wait for action server
        self.get_logger().info('Waiting for joint trajectory action server...')
        self.action_client.wait_for_server()
        self.get_logger().info('Connected to joint trajectory action server!')
        
        # Send initial position
        self.send_joint_positions()
    
    def degrees_to_radians(self, degrees):
        return degrees * (math.pi / 180.0)
    
    def clamp_joint(self, value_deg, joint_idx):
        """Clamp joint value within limits"""
        min_val, max_val = self.limits[joint_idx]
        return max(min_val, min(max_val, value_deg))
    
    def send_joint_positions(self):
        """Send current joint positions via action"""
        # Cancel any existing goal
        if self.current_goal_handle and not self.current_goal_handle.done():
            self.current_goal_handle.cancel()
        
        # Create goal message
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names
        
        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = [self.degrees_to_radians(pos) for pos in self.joint_positions]
        point.time_from_start = Duration(seconds=0, nanoseconds=int(self.movement_time * 1e9)).to_msg()
        
        goal_msg.trajectory.points = [point]
        goal_msg.goal_time_tolerance = Duration(seconds=1, nanoseconds=0).to_msg()
        
        # Send goal asynchronously
        self.current_goal_handle = self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        
        # Log current positions
        pos_str = ", ".join([f"{name}: {pos:5.1f}°" 
                           for name, pos in zip(self.joint_names, self.joint_positions)])
        self.get_logger().info(f"Moving to: [{pos_str}]")
    
    def feedback_callback(self, feedback_msg):
        """Handle action feedback (optional)"""
        pass
    
    def get_key(self):
        """Get a single character from stdin"""
        if not TERMIOS_AVAILABLE:
            return input("Enter command: ").strip()[:1]
        
        try:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                key = sys.stdin.read(1)
                if ord(key) == 3:  # Ctrl+C
                    raise KeyboardInterrupt
                return key
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception as e:
            self.get_logger().error(f"Input error: {e}")
            return 'q'
    
    def print_status(self):
        """Print current joint positions and controls"""
        if TERMIOS_AVAILABLE:
            print("\033[2J\033[H", end="")  # Clear screen
        else:
            print("\n" + "="*70)
        
        print("=" * 70)
        print("⚡ FAST ROS2 ROVER KEYBOARD CONTROLLER ⚡")
        print("=" * 70)
        for i, (name, pos) in enumerate(zip(self.joint_names, self.joint_positions)):
            key_map = ['A/a', 'S/s', 'D/d', 'F/f', 'G/g', 'H/h'][i]
            print(f"{name:20}: {pos:6.1f}° | {key_map}")
        print("=" * 70)
        print("CONTROLS:")
        print("  Uppercase: Positive | Lowercase: Negative")
        print("  R/r: Reset | Q/q: Quit | +/-: Change step size")
        print(f"  Step size: {self.step_size}° | Movement time: {self.movement_time}s")
        print("=" * 70)
    
    def handle_key(self, key):
        """Handle keyboard input"""
        if not key:
            return True
            
        key_lower = key.lower()
        old_positions = self.joint_positions.copy()
        
        # Joint controls mapping
        joint_keys = ['a', 's', 'd', 'f', 'g', 'h']
        
        if key_lower in joint_keys:
            joint_idx = joint_keys.index(key_lower)
            
            if key.isupper():
                new_pos = self.joint_positions[joint_idx] + self.step_size
            else:
                new_pos = self.joint_positions[joint_idx] - self.step_size
            
            # Apply limits
            self.joint_positions[joint_idx] = self.clamp_joint(new_pos, joint_idx)
            
            # Send new positions immediately
            self.send_joint_positions()
            
        elif key_lower == 'r':
            self.joint_positions = [0.0] * 6
            self.get_logger().info("🔄 Reset all joints to 0°")
            self.send_joint_positions()
        
        elif key == '+' or key == '=':
            self.step_size = min(45.0, self.step_size + 1.0)
            self.get_logger().info(f"📈 Step size: {self.step_size}°")
            
        elif key == '-' or key == '_':
            self.step_size = max(1.0, self.step_size - 1.0)
            self.get_logger().info(f"📉 Step size: {self.step_size}°")
            
        elif key_lower == 't':
            # Toggle movement speed
            if self.movement_time == 0.5:
                self.movement_time = 0.2  # Super fast
                self.get_logger().info("🚀 Super fast mode!")
            elif self.movement_time == 0.2:
                self.movement_time = 1.0  # Slow and smooth
                self.get_logger().info("🐌 Smooth mode!")
            else:
                self.movement_time = 0.5  # Normal
                self.get_logger().info("⚡ Normal speed!")
            
        elif key_lower == 'q':
            self.get_logger().info("👋 Quitting...")
            return False
            
        else:
            if not TERMIOS_AVAILABLE:
                print(f"Unknown command: {key}")
        
        return True
    
    def run_keyboard_loop(self):
        """Main keyboard control loop"""
        try:
            while self.running and rclpy.ok():
                self.print_status()
                
                if TERMIOS_AVAILABLE:
                    print("⌨️  Waiting for key press...")
                
                key = self.get_key()
                
                if not self.handle_key(key):
                    break
                    
                # Quick spin to handle ROS callbacks
                rclpy.spin_once(self, timeout_sec=0.01)
                    
        except KeyboardInterrupt:
            self.get_logger().info("Interrupted by user")
        except Exception as e:
            self.get_logger().error(f"Error in keyboard loop: {e}")
        
        # Cancel any active goals
        if self.current_goal_handle and not self.current_goal_handle.done():
            self.current_goal_handle.cancel()
        
        self.get_logger().info("Fast keyboard controller stopped.")

def main(args=None):
    rclpy.init(args=args)
    
    try:
        controller = FastRoverController()
        
        # Run ROS2 spinning in a separate thread
        spin_thread = threading.Thread(target=lambda: rclpy.spin(controller), daemon=True)
        spin_thread.start()
        
        # Run keyboard loop in main thread
        controller.run_keyboard_loop()
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()

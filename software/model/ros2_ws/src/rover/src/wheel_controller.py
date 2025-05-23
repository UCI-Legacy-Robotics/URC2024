import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import sys
import threading
import time
import math

try:
    import termios
    import tty
    TERMIOS_AVAILABLE = True
except ImportError:
    TERMIOS_AVAILABLE = False

class WheelVelocityController(Node):
    def __init__(self):
        super().__init__('wheel_velocity_controller')
        
        # Create publisher for wheel commands
        self.wheel_pub = self.create_publisher(
            Float64MultiArray, 
            '/mobile_base_controller/commands', 
            10
        )
        
        # Rover parameters
        self.wheel_radius = 0.1        # meters
        self.wheel_separation = 0.8    # distance between left/right wheels
        
        # Movement parameters
        self.linear_speed = 1.0     # m/s
        self.angular_speed = 1.0    # rad/s
        
        # Current wheel velocities [FL, FR, RL, RR]
        self.wheel_velocities = [0.0, 0.0, 0.0, 0.0]
        
        self.running = True
        
        # Publish commands at 20Hz
        self.timer = self.create_timer(0.05, self.publish_commands)
        
        self.get_logger().info('🚗 Direct Wheel Velocity Controller Started!')
        self.get_logger().info('Publishing to: /mobile_base_controller/commands')
    
    def calculate_wheel_velocities(self, linear_x, angular_z):
        """
        Calculate wheel velocities for differential drive
        Returns: [front_left, front_right, rear_left, rear_right] in rad/s
        """
        # Convert linear velocity to wheel angular velocity
        linear_wheel_vel = linear_x / self.wheel_radius
        
        # Convert angular velocity to differential wheel velocity
        angular_wheel_vel = angular_z * (self.wheel_separation / 2.0) / self.wheel_radius
        
        # Calculate individual wheel velocities
        # Left wheels (positive angular_z = turn left = left slower)
        left_vel = linear_wheel_vel - angular_wheel_vel
        # Right wheels (positive angular_z = turn left = right faster)
        right_vel = linear_wheel_vel + angular_wheel_vel
        
        return [left_vel, right_vel, left_vel, right_vel]
    
    def publish_commands(self):
        """Publish current wheel velocities"""
        msg = Float64MultiArray()
        msg.data = self.wheel_velocities
        self.wheel_pub.publish(msg)
    
    def stop_robot(self):
        """Stop all wheels"""
        self.wheel_velocities = [0.0, 0.0, 0.0, 0.0]
        self.get_logger().info("🛑 STOP")
    
    def get_key(self):
        """Get keyboard input"""
        if not TERMIOS_AVAILABLE:
            return input("Enter command (wasd/q): ").strip().lower()
        
        try:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                key = sys.stdin.read(1)
                if ord(key) == 3:  # Ctrl+C
                    raise KeyboardInterrupt
                # Handle arrow keys
                if ord(key) == 27:  # ESC
                    key = sys.stdin.read(1)
                    if key == '[':
                        key = sys.stdin.read(1)
                        if key == 'A': return 'w'    # Up arrow = forward
                        elif key == 'B': return 's'  # Down arrow = backward
                        elif key == 'C': return 'd'  # Right arrow = right
                        elif key == 'D': return 'a'  # Left arrow = left
                return key.lower()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            return 'q'
    
    def print_status(self):
        """Print current status"""
        if TERMIOS_AVAILABLE:
            print("\033[2J\033[H", end="")  # Clear screen
        else:
            print("\n" + "="*70)
        
        print("=" * 70)
        print("🚗 DIRECT WHEEL VELOCITY CONTROLLER 🚗")
        print("=" * 70)
        print("WHEEL VELOCITIES (rad/s):")
        print(f"  Front Left:  {self.wheel_velocities[0]:7.2f}")
        print(f"  Front Right: {self.wheel_velocities[1]:7.2f}")
        print(f"  Rear Left:   {self.wheel_velocities[2]:7.2f}")
        print(f"  Rear Right:  {self.wheel_velocities[3]:7.2f}")
        print("=" * 70)
        print("CONTROLS:")
        print("  W / ↑  : Forward")
        print("  S / ↓  : Backward")
        print("  A / ←  : Turn Left") 
        print("  D / →  : Turn Right")
        print("  SPACE  : Stop")
        print("  + / -  : Speed Up/Down")
        print("  Q      : Quit")
        print("=" * 70)
        print(f"Linear Speed:  {self.linear_speed:.1f} m/s")
        print(f"Angular Speed: {self.angular_speed:.1f} rad/s")
        print("=" * 70)
    
    def handle_key(self, key):
        """Handle keyboard input"""
        if key == 'w':  # Forward
            self.wheel_velocities = self.calculate_wheel_velocities(self.linear_speed, 0.0)
            self.get_logger().info("⬆️ FORWARD")
            
        elif key == 's':  # Backward
            self.wheel_velocities = self.calculate_wheel_velocities(-self.linear_speed, 0.0)
            self.get_logger().info("⬇️ BACKWARD")
            
        elif key == 'a':  # Turn left
            self.wheel_velocities = self.calculate_wheel_velocities(0.0, self.angular_speed)
            self.get_logger().info("⬅️ LEFT")
            
        elif key == 'd':  # Turn right
            self.wheel_velocities = self.calculate_wheel_velocities(0.0, -self.angular_speed)
            self.get_logger().info("➡️ RIGHT")
            
        elif key == ' ' or key == 'x':  # Stop
            self.stop_robot()
            
        elif key == '+' or key == '=':  # Speed up
            self.linear_speed = min(3.0, self.linear_speed + 0.2)
            self.angular_speed = min(3.0, self.angular_speed + 0.2)
            self.get_logger().info(f"📈 Speed: {self.linear_speed:.1f} m/s")
            
        elif key == '-':  # Speed down
            self.linear_speed = max(0.2, self.linear_speed - 0.2)
            self.angular_speed = max(0.2, self.angular_speed - 0.2)
            self.get_logger().info(f"📉 Speed: {self.linear_speed:.1f} m/s")
            
        elif key == 'r':  # Reset
            self.linear_speed = 1.0
            self.angular_speed = 1.0
            self.stop_robot()
            self.get_logger().info("🔄 RESET")
            
        elif key == 'q':  # Quit
            self.stop_robot()
            self.get_logger().info("👋 QUIT")
            return False
            
        else:
            self.stop_robot()  # Stop on unknown key for safety
            
        return True
    
    def run(self):
        """Main control loop"""
        try:
            while self.running and rclpy.ok():
                self.print_status()
                
                if TERMIOS_AVAILABLE:
                    print("🎮 Press keys to drive...")
                
                key = self.get_key()
                
                if not self.handle_key(key):
                    break
                    
                # Spin once to handle ROS callbacks
                rclpy.spin_once(self, timeout_sec=0.01)
                
        except KeyboardInterrupt:
            self.get_logger().info("Interrupted!")
        except Exception as e:
            self.get_logger().error(f"Error: {e}")
        
        self.stop_robot()
        time.sleep(0.1)  # Give time for stop command

def main():
    rclpy.init()
    
    try:
        controller = WheelVelocityController()
        
        # Run ROS in background thread
        ros_thread = threading.Thread(target=lambda: rclpy.spin(controller), daemon=True)
        ros_thread.start()
        
        # Run keyboard control in main thread
        controller.run()
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()

from tutorial_interfaces.srv import String

import rclpy
from rclpy.node import Node


class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(String, 'send_string', self.send_string_callback)

    def send_string_callback(self, request, response):
        response.response = "hello " + request.str
        self.get_logger().info('Incoming request\nstring: %s' % (request.str))

        return response


def main():
    rclpy.init()

    minimal_service = MinimalService()

    rclpy.spin(minimal_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()

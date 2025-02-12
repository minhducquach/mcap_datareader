import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Point, Quaternion
from std_msgs.msg import String
from builtin_interfaces.msg import Time
from mcap_plot.pose_module import Pose, Data

import numpy as np

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            PoseStamped,
            'light_source/pose',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.light = Data()

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg)
        # self.get_logger().info('I heard: "%s"' % msg.pose.position.x)
        pose = Pose(msg.pose.position, msg.pose.orientation)
        self.light.datapoints.append(pose)


def main(args=None):
    try:
        rclpy.init(args=args)
        minimal_subscriber = MinimalSubscriber()
        rclpy.spin(minimal_subscriber)
    except (KeyboardInterrupt, ExternalShutdownException):
        data_list = minimal_subscriber.light
        print(len(data_list.datapoints))


if __name__ == '__main__':
    main()
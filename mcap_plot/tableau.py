import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Point, Quaternion
from std_msgs.msg import String
from mcap_plot.pose_module import Pose, Data

import numpy as np

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            PoseStamped,
            # String,
            # '/topic',
            'tableau_blanc/pose',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.data_list = Data()

    def listener_callback(self, msg):
        # self.get_logger().info('I heard: "%s"' % msg.pose.position.x)
        pose = Pose(msg.pose.position, msg.pose.orientation)
        self.data_list.datapoints.append(pose)
        # print(self.data_list.datapoints)


def main(args=None):
    try:
        rclpy.init(args=args)
        minimal_subscriber = MinimalSubscriber()
        rclpy.spin(minimal_subscriber)
    except (KeyboardInterrupt, ExternalShutdownException):
        data_list = minimal_subscriber.data_list
        print(len(data_list.datapoints))


if __name__ == '__main__':
    main()
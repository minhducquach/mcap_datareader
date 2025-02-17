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
            'tableau_blanc/pose',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        # self.tableau = Data()
        self.file_1 = open("/home/manip/ros2_ws/src/mcap_plot/mcap_plot/tableau_pos_data.txt", "a")
        self.file_2 = open("/home/manip/ros2_ws/src/mcap_plot/mcap_plot/tableau_ori_data.txt", "a")

    def listener_callback(self, msg):
        # self.get_logger().info('I heard: "%s"' % msg.pose.position.x)
        # pose = Pose(msg.pose.position, msg.pose.orientation)
        # self.tableau.datapoints.append(pose)
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.file_1.write(f"{timestamp} {msg.pose.position.x} {msg.pose.position.y} {msg.pose.position.z} \n")
        self.file_1.flush()
        self.file_2.write(f"{timestamp} {msg.pose.orientation.x} {msg.pose.orientation.y} {msg.pose.orientation.z} {msg.pose.orientation.w} \n")
        self.file_2.flush()

    def close(self):
        self.file_1.close()
        self.file_2.close()


def main(args=None):
    try:
        rclpy.init(args=args)
        minimal_subscriber = MinimalSubscriber()
        rclpy.spin(minimal_subscriber)
    except (KeyboardInterrupt, ExternalShutdownException):
        # data_list = minimal_subscriber.data_list
        # print(len(data_list.datapoints))
        minimal_subscriber.close()


if __name__ == '__main__':
    main()
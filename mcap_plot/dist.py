import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Point, Quaternion
from std_msgs.msg import String
from builtin_interfaces.msg import Time
from mcap_plot.pose_module import Pose, Data

import numpy as np
from math import sqrt

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription_1 = self.create_subscription(
            PoseStamped,
            'light_source/pose',
            self.listener_callback,
            10)
        self.subscription_2 = self.create_subscription(
            PoseStamped,
            'tableau_blanc/pose',
            self.listener_callback_2,
            10)
        self.file_1 = open("/home/manip/ros2_ws/src/mcap_plot/mcap_plot/light_tab_dis_data.txt", "a")
        self.light = []
        self.tableau = []
        self.timer = self.create_timer(0.1, self.dist_calc)
        self.get_logger().info('MinimalSubscriber node has been started.')

    def listener_callback(self, msg):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.light.append([timestamp, msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.get_logger().info(f'Light source data received: {self.light[-1]}')

    def listener_callback_2(self, msg):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.tableau.append([timestamp, msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.get_logger().info(f'Tableau blanc data received: {self.tableau[-1]}')

    def dist_calc(self):
        if len(self.light) != 0 and len(self.tableau) != 0:
            if abs(self.light[0][0] - self.tableau[0][0]) <= 0.1: 
                dist = sqrt((self.light[0][1] - self.tableau[0][1]) ** 2 + 
                            (self.light[0][2] - self.tableau[0][2]) ** 2 + 
                            (self.light[0][3] - self.tableau[0][3]) ** 2)
                self.file_1.write(f"{self.light[0][0]} {dist}\n")
                self.file_1.flush()
                self.get_logger().info(f'Distance calculated and written to file: {dist}')
                self.light.pop(0)
                self.tableau.pop(0)

    def close(self):
        self.file_1.close()
        self.get_logger().info('Files have been closed.')

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    try:
        rclpy.spin(minimal_subscriber)
    except (KeyboardInterrupt, ExternalShutdownException):
        minimal_subscriber.close()
    finally:
        minimal_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Point, Quaternion
from std_msgs.msg import String
from builtin_interfaces.msg import Time
from mcap_plot.pose_module import Pose, Data

import numpy as np
from math import sqrt

import threading

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
        # self.light = Data()
        self.file_1 = open("/home/manip/ros2_ws/src/mcap_plot/mcap_plot/light_tab_dis_data.txt", "a")
        self.light = []
        self.tableau = []
        self.dist_calc()
        self.t1 = threading.Thread(target=self.dist_calc)
        self.t1.start()


    def listener_callback(self, msg):
        # self.get_logger().info('I heard: "%s"' % msg)
        # # self.get_logger().info('I heard: "%s"' % msg.pose.position.x)
        # pose = Pose(msg.pose.position, msg.pose.orientation)
        # self.light.datapoints.append(pose)
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        # self.file_1.write(f"{timestamp} {msg.pose.position.x} {msg.pose.position.y} {msg.pose.position.z} \n")
        # self.file_1.flush()
        self.light.append([timestamp, msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

    def listener_callback_2(self, msg):
        # self.get_logger().info('I heard: "%s"' % msg)
        # # self.get_logger().info('I heard: "%s"' % msg.pose.position.x)
        # pose = Pose(msg.pose.position, msg.pose.orientation)
        # self.light.datapoints.append(pose)
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        # self.file_1.write(f"{timestamp} {msg.pose.position.x} {msg.pose.position.y} {msg.pose.position.z} \n")
        # self.file_1.flush()
        self.tableau.append([timestamp, msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
    
    def close(self):
        self.t1.join()
        self.file_1.close()
        # self.file_2.close()
        # pass

    def dist_calc(self):
        # cmp1 = []
        # cmp2 = []
        # print("IN1", len(self.light), len(self.tableau))
        # if len(self.light) <= len(self.tableau):
        #     # print("IN")
        #     cmp1 = self.light
        #     cmp2 = self.tableau
        # else:
        #     # print("IN2")
        #     cmp1 = self.tableau
        #     cmp2 = self.light
        # i = 0
        # for item in cmp1:
        #     # print(i, cmp2[i][0])
        #     if abs(item[0] - cmp2[i][0]) <= 0.1:
        #         dist = sqrt((item[1] - cmp2[i][1]) * (item[1] - cmp2[i][1]) + (item[2] - cmp2[i][2]) * (item[2] - cmp2[i][2]) + (item[3] - cmp2[i][3]) * (item[3] - cmp2[i][3]))
        #         self.file_1.write(f"{item[0]} {dist}\n")
        #         self.file_1.flush()
        #     else:
        #         i += 1
        # self.close()

        while(1):
            if len(self.light) != 0 and len(self.tableau) != 0:
                if (abs(self.light[0][0] - self.tableau[0][0]) <= 0.1):
                    dist = sqrt((self.light[0][1] - self.tableau[0][1]) * (self.light[0][1] - self.tableau[0][1]) + (self.light[0][2] - self.tableau[0][2]) * (self.light[0][2] - self.tableau[0][2]) + (self.light[0][3] - self.tableau[0][3]) * (self.light[0][3] - self.tableau[0][3]))
                    self.file_1.write(f"{self.light[0][0]} {dist}\n")
                    self.file_1.flush()


def main(args=None):
    try:
        rclpy.init(args=args)
        minimal_subscriber = MinimalSubscriber()
        rclpy.spin(minimal_subscriber)
    except (KeyboardInterrupt, ExternalShutdownException):
        # data_list = minimal_subscriber.light
        # print(len(data_list.datapoints))
        minimal_subscriber.close()
        pass

if __name__ == '__main__':
    main()
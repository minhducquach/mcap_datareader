import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Point, Quaternion
from std_msgs.msg import String
from builtin_interfaces.msg import Time
from mcap_plot.pose_module import Pose, Data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

from message_filters import ApproximateTimeSynchronizer, Subscriber

import numpy as np
from math import sqrt

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        # self.subscription_1 = self.create_subscription(
        #     PoseStamped,
        #     'light_source/pose',
        #     self.listener_callback,
        #     10)
        # self.subscription_2 = self.create_subscription(
        #     PoseStamped,
        #     'tableau_blanc/pose',
        #     self.listener_callback_2,
        #     10)
        self.light_sub = Subscriber(self, PoseStamped, 'light_source/pose')
        self.tableau_sub = Subscriber(self, PoseStamped, 'tableau_blanc/pose')
        self.img_sub = Subscriber(self, Image, '/camera/color/image_raw')
        self.flag  = 0

        self.ts = ApproximateTimeSynchronizer([self.light_sub, self.tableau_sub, self.img_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.ts_callback)

        # self.image_subscriber = self.create_subscription(Image, 
        #                                                  "/camera/color/image_raw",
        #                                                  self.callback,
        #                                                  10)
        self.bridge = CvBridge()
        # self.file = open("/home/manip/ros2_ws/src/mcap_plot/mcap_plot/image_data.txt", "a")
        self.file_1 = open("/home/manip/ros2_ws/src/mcap_plot/mcap_plot/light_tab_dis_data.txt", "a")
        # self.light = []
        # self.tableau = []
        # self.timer = self.create_timer(0.05, self.dist_calc)
        # self.get_logger().info('MinimalSubscriber node has been started.')

    def ts_callback(self, light_msg, tableau_msg, img_msg):
        # Dist
        if (light_msg.header.stamp.sec > 1738589914):
            dist = sqrt((light_msg.pose.position.x - tableau_msg.pose.position.x) ** 2 + 
                                (light_msg.pose.position.y - tableau_msg.pose.position.y) ** 2 + 
                                (light_msg.pose.position.z - tableau_msg.pose.position.z) ** 2)
            
            # Img
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "rgb8")
            if self.flag == 0:
                cv2.imwrite('/home/manip/ros2_ws/src/mcap_plot/mcap_plot/test_img.png', cv_image)
                self.flag = 1
            (rows, cols, channels) = cv_image.shape
            center_pixel = cv_image[rows//2, cols//2].tolist()
            intensity = center_pixel[0] * 0.2126 + center_pixel[1] * 0.7152 + center_pixel[2] * 0.0722

            self.file_1.write(f"{dist} {int(intensity)}\n")
            self.file_1.flush()

    # def listener_callback(self, msg):
    #     timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    #     self.light.append([timestamp, msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
    #     # self.get_logger().info(f'Light source data received: {self.light[-1]}')

    # def listener_callback_2(self, msg):
    #     timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    #     self.tableau.append([timestamp, msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        # self.get_logger().info(f'Tableau blanc data received: {self.tableau[-1]}')

    # def callback(self, data):
    #     try:
    #         cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
    #     except CvBridgeError as e:
    #         print(e)
    #     (rows, cols, channels) = cv_image.shape
    #     center_pixel = cv_image[rows//2, cols//2].tolist()
    #     intensity = center_pixel[0] * 0.2126 + center_pixel[1] * 0.7152 + center_pixel[2] * 0.0722
    #     timestamp = data.header.stamp.sec + data.header.stamp.nanosec * 1e-9
    #     self.file.write(f"{timestamp} {int(intensity)}\n")
    #     self.file.flush()

    # def dist_calc(self):
    #     if len(self.light) != 0 and len(self.tableau) != 0:
    #         print("IN", abs(self.light[0][0] - self.tableau[0][0]))
    #         if abs(self.light[0][0] - self.tableau[0][0]) <= 0.1: 
    #             dist = sqrt((self.light[0][1] - self.tableau[0][1]) ** 2 + 
    #                         (self.light[0][2] - self.tableau[0][2]) ** 2 + 
    #                         (self.light[0][3] - self.tableau[0][3]) ** 2)
    #             self.file_1.write(f"{self.light[0][0]} {dist}\n")
    #             self.file_1.flush()
    #             # self.get_logger().info(f'Distance calculated and written to file: {dist}')
    #             self.light.pop(0)
    #             self.tableau.pop(0)

    def close(self):
        self.file_1.close()
        # self.get_logger().info('Files have been closed.')

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
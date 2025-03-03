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

from math import cos

from scipy.spatial.transform import Rotation
from message_filters import ApproximateTimeSynchronizer, Subscriber

import numpy as np
from math import sqrt
import pyquaternion as pyq

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

        self.ts = ApproximateTimeSynchronizer([self.light_sub, self.tableau_sub, self.img_sub], queue_size=10, slop=0.03)
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
        self.init_or = None

    def ts_callback(self, light_msg, tableau_msg, img_msg):
        # Dist
            # print(tableau_msg)
        if (light_msg.header.stamp.sec > 1738589914):
            vt = np.array([light_msg.pose.position.x - tableau_msg.pose.position.x, light_msg.pose.position.y - tableau_msg.pose.position.y, light_msg.pose.position.z - tableau_msg.pose.position.z])
            dist = sqrt((light_msg.pose.position.x - tableau_msg.pose.position.x) ** 2 + 
                                (light_msg.pose.position.y - tableau_msg.pose.position.y) ** 2 + 
                                (light_msg.pose.position.z - tableau_msg.pose.position.z) ** 2)
            # o_tab = np.array([tableau_msg.pose.orientation.x, tableau_msg.pose.orientation.y, tableau_msg.pose.orientation.z])
            # o_light = np.array([light_msg.pose.orientation.x, light_msg.pose.orientation.y, light_msg.pose.orientation.z, light_msg.pose.orientation.w])
            
            # alpha: vector of pos
            cos_angle_alpha = np.dot(vt, np.array([-1,0,0]))/(np.linalg.norm(vt) * np.linalg.norm(np.array([-1,0,0])))

            # beta
            norm_vec = np.cross(vt, np.array([-1,0,0]))  # calc norm_vec of plane between pos vect and vector -x
            q_light = pyq.Quaternion(axis=[light_msg.pose.orientation.x, light_msg.pose.orientation.y, light_msg.pose.orientation.z], angle=light_msg.pose.orientation.w)
            light_dir = q_light.rotate(np.array([1.0, 0.0, 0.0]))

            proj_dir = norm_vec * np.dot(light_dir, norm_vec) / np.dot(norm_vec, norm_vec) # proj light_dir to the norm_vec of plane
            proj_dir = light_dir - proj_dir

            cos_angle_beta = np.dot(vt, proj_dir)/(np.linalg.norm(vt) * np.linalg.norm(proj_dir))
            # cos_angle_beta = np.dot(vt, light_dir)/(np.linalg.norm(vt) * np.linalg.norm(light_dir))
            # rot = Rotation.from_quat(o_light)
            # angles = rot.as_euler('xyz', degrees=True)

            # 3 angles
            # ypr = q_light.yaw_pitch_roll

            # test_idea
            # q_light = pyq.Quaternion(axis=[light_msg.pose.orientation.x, light_msg.pose.orientation.y, light_msg.pose.orientation.z], angle=light_msg.pose.orientation.w)
            light_dir = q_light.rotate(np.array([1.0, 0.0, 0.0]))
            cos_angle_y = np.dot(np.array([1,0,0]), light_dir)/(np.linalg.norm(np.array([1,0,0])) * np.linalg.norm(light_dir))

            # cos_angle_y = np.dot(np.array([1,0,0]), proj_dir)/(np.linalg.norm(np.array([1,0,0])) * np.linalg.norm(proj_dir))

            light_dir = q_light.rotate(np.array([0.0, 1.0, 0.0]))
            cos_angle_p = np.dot(np.array([0,1,0]), light_dir)/(np.linalg.norm(np.array([1,0,0])) * np.linalg.norm(light_dir))

            light_dir = q_light.rotate(np.array([0.0, 0.0, 1.0]))
            cos_angle_r = np.dot(np.array([0,0,1]), light_dir)/(np.linalg.norm(np.array([1,0,0])) * np.linalg.norm(light_dir))

            # Img
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "rgb8")
            if self.flag == 0:
                cv2.imwrite('/home/manip/ros2_ws/src/mcap_plot/mcap_plot/test_img.png', cv_image)
                self.flag = 1
            (rows, cols, channels) = cv_image.shape
            center_pixel = cv_image[rows//2, cols//2].tolist()
            intensity = center_pixel[0] * 0.2126 + center_pixel[1] * 0.7152 + center_pixel[2] * 0.0722

            self.file_1.write(f"{dist} {int(intensity)} {cos_angle_alpha} {cos_angle_beta} {cos_angle_y} {cos_angle_p} {cos_angle_r}\n")
            # self.file_1.write(f"{dist} {int(intensity)} {cos_angle_alpha} {cos_angle_beta} {cos(ypr[2])} {cos(ypr[1])} {cos(ypr[0])}\n")
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
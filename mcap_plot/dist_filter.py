import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Point, Quaternion
from std_msgs.msg import String
from builtin_interfaces.msg import Time
# from mcap_plot.pose_module import Pose, Data
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
# import cv2.cv as cv
import apriltag

import mcap_plot.transform as transform
from math import cos

from scipy.spatial.transform import Rotation
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
        # #     10)
        # self.subscription_2 = self.create_subscription(
        #     PoseStamped,
        #     'tableau_blanc/pose',
        #     self.listener_callback_2,
        #     10)
        self.light_sub = Subscriber(self, PoseStamped, 'Lamp/pose')
        self.tableau_sub = Subscriber(self, PoseStamped, 'tableau_blanc/pose')
        self.imu_sub = Subscriber(self, PoseStamped, '/Darknav/pose')
        self.img_sub = Subscriber(self, CompressedImage, 'camera/color/image_raw/compressed')
        self.img_depth_sub = Subscriber(self, Image, 'camera/depth/image_rect_raw')
        self.flag  = 0

        self.ts = ApproximateTimeSynchronizer([self.light_sub, self.tableau_sub, self.imu_sub, self.img_sub, self.img_depth_sub], queue_size=10, slop=0.03)
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

    def increase_brightness(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV color space
        h, s, v = cv2.split(hsv)  # Split channels

        # Apply CLAHE (Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        v = clahe.apply(v)

        # Merge back and convert to RGB
        hsv = cv2.merge((h,s,v))
        enhanced_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return enhanced_img

    def adjust_gamma(self, image, gamma=1.5):
        """Apply gamma correction to brighten darker images while keeping contrast."""
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)

    
    def center_getter(self, img):
        # img = self.increase_brightness(img)  # Improve brightness
        # img = self.adjust_gamma(img, 2)  # Apply gamma correction
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        options = apriltag.DetectorOptions(
            families="tag36h11", 
            refine_edges=True,  
            quad_decimate=1.0,  
            quad_blur=0.0,     
        )
        detector = apriltag.Detector(options)

        results = detector.detect(gray)
        c = [0, 0]

        if len(results) == 0:
            # print("No AprilTags detected")
            return c
        print((results))
        for r in results:
            c[0] += r.center[0]
            c[1] += r.center[1]
        
        c[0] = int(c[0] / len(results))
        c[1] = int(c[1] / len(results))

        return c


    def ts_callback(self, light_msg, tableau_msg, imu_msg, img_msg, img_depth_msg):
        # Dist
        # print("INs")
        # print(light_msg)
        vt = np.array([light_msg.pose.position.x - tableau_msg.pose.position.x, light_msg.pose.position.y - tableau_msg.pose.position.y, light_msg.pose.position.z - tableau_msg.pose.position.z])
        # print(f'vt: {vt}')
        dist = sqrt((light_msg.pose.position.x - tableau_msg.pose.position.x) ** 2 + 
                            (light_msg.pose.position.y - tableau_msg.pose.position.y) ** 2 + 
                            (light_msg.pose.position.z - tableau_msg.pose.position.z) ** 2)
        # o_tab = np.array([tableau_msg.pose.orientation.x, tableau_msg.pose.orientation.y, tableau_msg.pose.orientation.z])
        # o_light = np.array([light_msg.pose.orientation.x, light_msg.pose.orientation.y, light_msg.pose.orientation.z, light_msg.pose.orientation.w])
        
        # alpha: vector of pos
        board_orientation = [tableau_msg.pose.orientation.x, tableau_msg.pose.orientation.y, tableau_msg.pose.orientation.z, tableau_msg.pose.orientation.w]
        # print(board_orientation[3])
        board_norm = transform.rotate_vec(np.array([0,0,1]), board_orientation)
        # print(board_norm)

        # cos_angle_alpha = np.dot(vt, board_norm)/(np.linalg.norm(vt))
        cos_angle_alpha = np.dot(vt, board_norm)
        # print(cos_angle_alpha)


        # angle between ray and centerline
        light_orientation = [light_msg.pose.orientation.x, light_msg.pose.orientation.y, light_msg.pose.orientation.z, light_msg.pose.orientation.w]
        light_normal = transform.rotate_vec(np.array([0,0,1]), light_orientation)
        cos_angle_beta = np.dot(-vt, light_normal)/np.linalg.norm(vt)

        # Img
        cv_image = self.bridge.compressed_imgmsg_to_cv2(img_msg, "rgb8")
        depth_image = self.bridge.imgmsg_to_cv2(img_depth_msg, "32FC1")
        depth_array = np.array(depth_image, dtype=np.float32).reshape(480, 848)
        # print(depth_array.size)

        cv_image = self.increase_brightness(cv_image)  # Improve brightness
        cv_image = self.adjust_gamma(cv_image, 3)  # Apply gamma correction

        # center = self.center_getter(cv_image)
        # cv_image = self.increase_brightness(cv_image, 50)
        # Intrinsics
        fx = 432.3138427734375
        fy = 432.3138427734375
        cx = 418.7983703613281
        cy = 240.33827209472656

        # Original pixel locations of tags
        tag_centers_2d = [
            [495.24536768, 173.51998179],
            [373.64785926, 238.94726765],
            [495.91353533, 238.49124637]
        ]

        tag_centers_3d = []

        # Convert to 3D
        for u, v in tag_centers_2d:
            u_int = int(round(u))
            v_int = int(round(v))
            Z = depth_array[v_int, u_int]
            
            if Z == 0 or np.isnan(Z):
                print(f"Invalid depth at ({u_int}, {v_int})")
                continue  # Or handle accordingly

            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            tag_centers_3d.append([X, Y, Z])

        # Make sure we got 3 valid points
        if len(tag_centers_3d) < 3:
            raise ValueError("Not enough valid tag points for normal calculation")

        # Vectors in 3D
        tag_vt_1 = np.array(tag_centers_3d[0]) - np.array(tag_centers_3d[2])
        tag_vt_2 = np.array(tag_centers_3d[1]) - np.array(tag_centers_3d[2])

        # Plane normal
        board_norm_cam = np.cross(tag_vt_1, tag_vt_2)
        board_norm_cam = board_norm_cam / np.linalg.norm(board_norm_cam)
        print("Board normal vector (camera frame):", board_norm_cam)

        # View direction (from camera to center point)
        u_center, v_center = 428, 205
        Z_center = depth_array[v_center, u_center]
        X_center = (u_center - cx) * Z_center / fx
        Y_center = (v_center - cy) * Z_center / fy

        view_dir = np.array([-X_center, -Y_center, -Z_center])
        view_dir = view_dir / np.linalg.norm(view_dir)
        print("View vector (camera frame):", view_dir)

        # Cosine of angle between view direction and normal
        # cos_angle_view = np.dot(view_dir, board_norm_cam)
        # print("Cosine of view angle:", cos_angle_view)

        q_cam_imu = [-0.504111, 0.505896, -0.50522, 0.484452]
        q_imu_w = [imu_msg.pose.orientation.x, imu_msg.pose.orientation.y, imu_msg.pose.orientation.z, imu_msg.pose.orientation.w]
        view_dir = transform.rotate_vec(view_dir, q_cam_imu)
        view_dir = transform.rotate_vec(view_dir, q_imu_w)
        print("View vector (world frame):", view_dir)

        bisector = (view_dir + vt)/np.linalg.norm(view_dir + vt)
        blinnTerm = np.dot(board_norm, bisector)
        print("Blinn:", blinnTerm)
        

        if self.flag == 0:
            cv2.circle(cv_image, (428, 205), 5, (0, 0, 255), -1)
            cv2.imwrite('/home/manip/ros2_ws/src/mcap_plot/mcap_plot/test_img.png', cv_image)
            self.flag = 1
        # (rows, cols, channels) = cv_image.shape
        # center_pixel = cv_image[rows//2, cols//2].tolist()
        # print(center[1], center[0])
        center_pixel = cv_image[205, 428].tolist()
        # intensity = center_pixel[0] * 0.2126 + center_pixel[1] * 0.7152 + center_pixel[2] * 0.0722
        intensity = (center_pixel[0] + center_pixel[1] + center_pixel[2])/3

        # self.file_1.write(f"{dist} {int(intensity)} {cos_angle_alpha} {cos_angle_beta} {cos_angle_y} {cos_angle_p} {cos_angle_r}\n")
        # self.file_1.write(f"{dist} {int(intensity)} {cos_angle_alpha} {cos_angle_beta} {cos(ypr[2])} {cos(ypr[1])} {cos(ypr[0])}\n")
        # print(f"{dist} {int(intensity)} {cos_angle_alpha}\n")
        self.file_1.write(f"{dist} {intensity} {cos_angle_alpha} {cos_angle_beta} {blinnTerm}\n")
        self.file_1.flush()

    # def listener_callback(self, msg):
    #     timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    #     self.light.append([timestamp, msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
    #     # self.get_logger().info(f'Light source data received: {self.light[-1]}')

    def listener_callback_2(self, msg):
        # timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        # self.tableau.append([timestamp, msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        # self.get_logger().info(f'Tableau blanc data received: {self.tableau[-1]}')
        # print(msg)
        pass

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
    # print("IN")
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
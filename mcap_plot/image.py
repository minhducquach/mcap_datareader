import sys
import rclpy
import cv2

from rclpy.executors import ExternalShutdownException
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image

class ImageLogger(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.image_subscriber = self.create_subscription(Image, 
                                                         "/camera/color/image_raw",
                                                         self.callback,
                                                         10)
        self.bridge = CvBridge()
        self.file = open("image_data.txt", "a")

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            print(e)
        (rows, cols, channels) = cv_image.shape
        center_pixel = cv_image[rows//2, cols//2]
        timestamp = data.header.stamp.sec + data.header.stamp.nanosec * 1e-9
        self.file.write(f"{timestamp}, {center_pixel.tolist()}\n")
        self.file.flush()
        self.get_logger().info(f"Logged: {timestamp}, {center_pixel}")

    def close(self):
        self.file.close()


def main(args=None):
    try:
        rclpy.init(args=args)
        minimal_subscriber = ImageLogger()
        rclpy.spin(minimal_subscriber)
    except (KeyboardInterrupt, ExternalShutdownException):
        # data_list = minimal_subscriber.light
        # print(len(data_list.datapoints))
        # pass
        minimal_subscriber.close()
        


if __name__ == '__main__':
    main()

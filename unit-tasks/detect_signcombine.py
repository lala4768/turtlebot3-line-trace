#!/usr/bin/env python3
# Unified detect_signcombine.py for both parking and construction

from enum import Enum
import os
import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import UInt8

class DetectSignCombine(Node):
    def __init__(self):
        super().__init__('detect_signcombine')

        self.sub_image_type = 'raw'
        self.pub_image_type = 'compressed'

        if self.sub_image_type == 'compressed':
            self.sub_image_original = self.create_subscription(
                CompressedImage,
                '/detect/image_input/compressed',
                self.cbFindTrafficSign,
                10
            )
        elif self.sub_image_type == 'raw':
            self.sub_image_original = self.create_subscription(
                Image,
                '/detect/image_input',
                self.cbFindTrafficSign,
                10
            )

        self.pub_traffic_sign = self.create_publisher(UInt8, '/detect/traffic_sign', 10)
        if self.pub_image_type == 'compressed':
            self.pub_image_traffic_sign = self.create_publisher(
                CompressedImage,
                '/detect/image_output/compressed', 10
            )
        elif self.pub_image_type == 'raw':
            self.pub_image_traffic_sign = self.create_publisher(
                Image, '/detect/image_output', 10
            )

        self.cvBridge = CvBridge()
        self.TrafficSign = Enum('TrafficSign', 'parking construction tunnel')
        self.counter = 1

        self.fnPreproc()
        self.get_logger().info('DetectSignCombine Node Initialized')

    def fnPreproc(self):
        self.sift = cv2.SIFT_create()
        dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'image')

        self.templates = {
            'parking': {
                'image': cv2.imread(os.path.join(dir_path, 'parking.png'), 0),
                'enum': self.TrafficSign.parking.value
            },
            'construction': {
                'image': cv2.imread(os.path.join(dir_path, 'construction.png'), 0),
                'enum': self.TrafficSign.construction.value
            },
            'tunnel': {
                'image': cv2.imread(os.path.join(dir_path, 'tunnel.png'), 0),
                'enum': self.TrafficSign.tunnel.value
            }

        }

        for key, value in self.templates.items():
            kp, des = self.sift.detectAndCompute(value['image'], None)
            value['kp'] = kp
            value['des'] = des

        FLANN_INDEX_KDTREE = 0
        index_params = { 'algorithm': FLANN_INDEX_KDTREE, 'trees': 5 }
        search_params = { 'checks': 50 }
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def fnCalcMSE(self, arr1, arr2):
        return np.sum((arr1 - arr2) ** 2) / (arr1.shape[0] * arr1.shape[1])

    def cbFindTrafficSign(self, image_msg):
        if self.counter % 3 != 0:
            self.counter += 1
            return
        else:
            self.counter = 1

        if self.sub_image_type == 'compressed':
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            cv_image_input = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        elif self.sub_image_type == 'raw':
            cv_image_input = self.cvBridge.imgmsg_to_cv2(image_msg, 'bgr8')

        kp1, des1 = self.sift.detectAndCompute(cv_image_input, None)
        if des1 is None or len(kp1) == 0:
            return

        MIN_MATCH_COUNT = 8
        MIN_MSE_DECISION = 50000

        for name, data in self.templates.items():
            matches = self.flann.knnMatch(des1, data['des'], k=2)
            good = [m for m, n in matches if m.distance < 0.7 * n.distance]
            #self.get_logger().info(f"{name} - Good matches: {len(good)}")

            if len(good) > MIN_MATCH_COUNT:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([data['kp'][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                mse = self.fnCalcMSE(src_pts, dst_pts)

                if mse < MIN_MSE_DECISION:
                    msg = UInt8()
                    msg.data = data['enum']
                    self.pub_traffic_sign.publish(msg)
                    self.get_logger().info(f"Detected sign: {name}")

                    final_img = cv2.drawMatches(
                        cv_image_input,
                        kp1,
                        data['image'],
                        data['kp'],
                        good,
                        None,
                        matchColor=(255, 0, 0),
                        singlePointColor=None,
                        matchesMask=mask.ravel().tolist() if mask is not None else None,
                        flags=2
                    )

                    if self.pub_image_type == 'compressed':
                        self.pub_image_traffic_sign.publish(
                            self.cvBridge.cv2_to_compressed_imgmsg(final_img, 'jpg')
                        )
                    elif self.pub_image_type == 'raw':
                        self.pub_image_traffic_sign.publish(
                            self.cvBridge.cv2_to_imgmsg(final_img, 'bgr8')
                        )
                    break

        # fallback to raw display if nothing matched
        if self.pub_image_type == 'compressed':
            self.pub_image_traffic_sign.publish(
                self.cvBridge.cv2_to_compressed_imgmsg(cv_image_input, 'jpg')
            )
        elif self.pub_image_type == 'raw':
            self.pub_image_traffic_sign.publish(
                self.cvBridge.cv2_to_imgmsg(cv_image_input, 'bgr8')
            )

def main(args=None):
    rclpy.init(args=args)
    node = DetectSignCombine()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

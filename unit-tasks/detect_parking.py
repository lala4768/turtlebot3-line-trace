#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import time
from enum import Enum
import math

from geometry_msgs.msg import Twist
from std_msgs.msg import UInt8
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

class DetectParking(Node):
    def __init__(self):
        super().__init__('detect_parking')

        self.parking_active = False
        self.construction_detected = False

        self.cmd_source_pub = self.create_publisher(String, '/mode/cmd_source', 10)
        self.pub_cmd_vel = self.create_publisher(Twist, '/parking/cmd_vel', 10)
        self.create_subscription(UInt8, '/detect/traffic_sign', self.cb_traffic_sign, 10)
        self.create_subscription(LaserScan, '/scan', self.cb_scan_obstacle, 10)

        self.StepOfParking = Enum('StepOfParking', 'parking construction')

        self.start_obstacle_detection = False
        self.obstacle_check_done = False

    def cb_traffic_sign(self, msg):
        if msg.data == self.StepOfParking.parking.value and not self.parking_active:
            self.get_logger().info('[Sign] Parking sign detected! Entering parking mode.')
            self.parking_active = True
            self.cmd_source_pub.publish(String(data='parking'))
            self.do_initial_parking_motion()

        elif self.parking_active and msg.data == self.StepOfParking.construction.value and not self.construction_detected:
            self.get_logger().info('[Sign] Construction sign detected! Performing side parking.')
            self.construction_detected = True
            self.park_on_side('right')  # or 'left' depending on your setup

    def move(self, linear_x, angular_z, duration, description):
        self.get_logger().info(f"[Parking] {description}")
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.pub_cmd_vel.publish(twist)
        time.sleep(duration)
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.pub_cmd_vel.publish(twist)
        time.sleep(1)

    def do_initial_parking_motion(self):
        self.get_logger().info("[Parking] Initial motion start")
        time.sleep(1)

        self.move(0.1, 0.0, 5.0, "Going forward into spot")
        self.move(0.0, 1.0, 1.6, "Turning left")
        self.move(0.1, 0.0, 8.3, "Aligning forward")


    def park_on_side(self, direction):
        ang = 1.0 if direction == 'right' else -1.0
        self.move(0.1, 0.0, 2.0, "Going Forward")
        self.move(0.0, ang, 1.6, f"Turning {direction}")
        self.move(0.1, 0.0, 2.0, "Going into spot")
        self.move(-0.1, 0.0, 2.0, "Backing out")
        self.move(0.0, ang, 1.6, f"Turning {direction} again")
        self.move(0.1, 0.0, 9.3, "Going Forward")
        self.move(0.0, ang, 1.6, "Final turn")
        self.move(0.1, 0.0, 2.0, "Re-align forward")

        self.cmd_source_pub.publish(String(data='lane'))
        self.get_logger().info("[Parking] Side parking complete. Returning to lane mode.")

    def cb_scan_obstacle(self, scan):
        # You can optionally use this if you want obstacle-based decision again
        pass

def main(args=None):
    rclpy.init(args=args)
    node = DetectParking()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

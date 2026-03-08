#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from enum import Enum

from std_msgs.msg import UInt8
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import time


class DetectStopTunnel(Node):
    def __init__(self):
        super().__init__('detect_stop_tunnel')

        self.pub_cmd_vel = self.create_publisher(Twist, '/tunnel/cmd_vel', 10)
        self.cmd_source_pub = self.create_publisher(String, '/mode/cmd_source', 10)  # ✅ 추가

        self.create_subscription(UInt8, '/detect/traffic_sign', self.cb_traffic_sign, 10)

        self.TrafficSign = Enum('TrafficSign', 'parking construction tunnel')

        self.stopped = False

    def cb_traffic_sign(self, msg):
        if msg.data == self.TrafficSign.tunnel.value and not self.stopped:
            self.get_logger().info('[Sign] 🛑 Tunnel sign detected! Switching to tunnel mode and stopping.')
            self.cmd_source_pub.publish(String(data='tunnel'))  # ✅ source 변경
            self.stop_robot()
            self.stopped = True

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        for _ in range(2):  # 여러 번 publish 해서 확실히 멈추게
            self.pub_cmd_vel.publish(twist)
            time.sleep(0.001)

def main(args=None):
    rclpy.init(args=args)
    node = DetectStopTunnel()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

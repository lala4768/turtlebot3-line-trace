#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class ControlTrafficLight(Node):
    def __init__(self):
        super().__init__('control_traffic_light')
        self.subscription = self.create_subscription(
            String,
            '/traffic_light_color',
            self.listener_callback,
            10
        )
        self.publisher = self.create_publisher(Twist, '/traffic_light/cmd_vel', 10)
        self.cmd_source_pub = self.create_publisher(String, '/mode/cmd_source', 10)
        self.traffic_light_count = 0
        self.current_color = 'red'
        self.previous_color = None
        self.light_detected_time = None
        self.cmd_source = 'lane'
        self.twist = Twist()
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.lane_entered_time = None  # ✅ 추가: lane 모드 진입 시간 저장용

    def listener_callback(self, msg):
        self.current_color = msg.data.lower().strip()

        # ✅ lane 모드로 들어간지 5초가 지나면 traffic_light 진입 차단
        if self.lane_entered_time:
            elapsed_since_lane = (self.get_clock().now() - self.lane_entered_time).nanoseconds / 1e9
            if elapsed_since_lane >= 5.0:
                self.get_logger().info(':⛔: lane 모드 5초 경과 → traffic_light 모드 진입 차단')
                return

        if self.current_color != self.previous_color:
            self.previous_color = self.current_color
            self.traffic_light_count += 1

            # 첫/두 번째 인식은 무시
            if self.traffic_light_count <= 0:
                self.get_logger().info(f':경고: [무시됨] {self.traffic_light_count}번째 신호등 감지: {self.current_color}')
                return

            # 세 번째 인식부터 동작
            self.set_cmd_source('traffic_light')
            self.light_detected_time = self.get_clock().now()
            if self.current_color == 'green':
                self.twist.linear.x = 0.15
                self.get_logger().info(':흰색_확인_표시: 초록불 인식 → 주행 시작 ')
            elif self.current_color == 'yellow':
                self.twist.linear.x = 0.03
                self.get_logger().info(':경고: 노랑불 인식 → 속도 감소 ')
            elif self.current_color == 'red':
                self.twist.linear.x = 0.0
                self.get_logger().info(':팔각형_기호: 빨간불 인식 → 정지 ')
            else:
                self.get_logger().warn(f':질문: 알 수 없는 색상: {self.current_color}')

    def timer_callback(self):
        if self.cmd_source == 'traffic_light':
            self.publisher.publish(self.twist)
            self.get_logger().info(f':자동차: 현재 속도: {self.twist.linear.x:.2f} m/s')
            if self.light_detected_time:
                elapsed = (self.get_clock().now() - self.light_detected_time).nanoseconds / 1e9
                if elapsed >= 4:
                    self.set_cmd_source('lane')
                    self.light_detected_time = None

    def set_cmd_source(self, source: str):
        if self.cmd_source != source:
            self.cmd_source = source
            self.cmd_source_pub.publish(String(data=source))
            self.get_logger().info(f':반복: cmd_source 전환 → {source}')
            if source == 'lane':
                self.lane_entered_time = self.get_clock().now()  # ✅ lane 진입 시각 저장

def main(args=None):
    rclpy.init(args=args)
    node = ControlTrafficLight()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

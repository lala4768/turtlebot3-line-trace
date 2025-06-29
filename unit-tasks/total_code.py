import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState, Image
from geometry_msgs.msg import Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge
import cv2
import numpy as np
from aruco_msgs.msg import MarkerArray
import time
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
class LaneFollower(Node):
    def __init__(self):
        super().__init__('lane_follower')
        params = {
            'y_h': (15,45), 'y_s': (50,255), 'y_v': (100,255),
            'w_h': (0,180),  'w_s': (0,40),   'w_v': (200,255),
            'pixel_thresh':150, 'recover_wait':3.0, 'recover_spin':5.0,
            'kp':0.006, 'ki':0.0, 'kd':0.01, 'i_max':1000.0,
            'max_v':0.05, 'min_v':0.03, 'max_a':1.0
        }
        for k,v in params.items():
            self.declare_parameter(k, v)
            setattr(self, k, self.get_parameter(k).value)
        # joint_states + arm command
        self.br = CvBridge(); self.frame=None
        self.joints=[0.0]*4; self.joint_names=['joint1','joint2','joint3','joint4']
        self.offset=0.45; self.state='restored'; self.recover=False
        self.prev_err=self.integral=0.0
        # 콜백/퍼블리셔
        self.create_subscription(CompressedImage, '/camera/image_raw/compressed', self.cb_image, 10)
        self.create_subscription(JointState, '/joint_states', self.cb_joint, 10)
        self.cmd_pub=self.create_publisher(Twist,'/cmd_vel',10)
        self.debug_pub=self.create_publisher(Image,'/lane/debug_frame',10)
        self.arm_pub=self.create_publisher(JointTrajectory,'/arm_controller/joint_trajectory',10)
        self.create_timer(0.05, self.loop)
        # pick_and_place trigger
        self.pick_pub = self.create_publisher(MarkerArray, '/pick_trigger', 10)
        self.marker_detected = False
        # 여기에 callback group 추가
        self.callback_group = MutuallyExclusiveCallbackGroup()
        # delayed stop 타이머를 위한 기본값
        self.delayed_stop_timer = None
        # PID params
        self.kp, self.ki, self.kd = 0.006, 0.0, 0.01
        self.prev_error = 0.0
        self.integral = 0.0
        self.max_a = 0.7
        self.min_v, self.max_v = 0.03, 0.05
        self.pick_in_progress = False
    def cb_image(self, msg):
        try: self.frame = self.br.compressed_imgmsg_to_cv2(msg,'bgr8')
        except: pass
    def cb_joint(self, msg):
        for n,p in zip(msg.name,msg.position):
            if n in self.joint_names: self.joints[self.joint_names.index(n)] = p
    def move_joint(self,pos):
        self.joints[0]=pos
        traj=JointTrajectory(joint_names=self.joint_names)
        pt=JointTrajectoryPoint(positions=self.joints,velocities=[0]*4,accelerations=[0]*4)
        pt.time_from_start.sec=1; traj.points=[pt]; self.arm_pub.publish(traj)
    def cmd(self,lin,ang=0.0):
        t=Twist(); t.linear.x=float(lin); t.angular.z=float(ang); self.cmd_pub.publish(t)
    def marker_callback(self, msg):
        for marker in msg.markers:
            if marker.id == 1:  # ← 원하는 ID
                #time.sleep(20.0)
                self.get_logger().info("Aruco Marker Detected! ID 1")
                self.detected_marker_array=msg
                #self.pick_pub.publish(marker)
                self.marker_detected = True
                break
    def delayed_stop(self):
        self.cmd_pub.publish(Twist())
        self.get_logger().info("Stopped after 20000 seconds delay.")
        if self.delayed_stop_timer is not None:
            self.delayed_stop_timer.cancel()
            self.delayed_stop_timer = None
    def image_callback(self, msg):
        # ArUco 마커가 감지된 경우
        if self.pick_in_progress:
            return
        if self.marker_detected:
            self.get_logger().info("Aruco Marker detected - stopping robot and triggering pick task.")
            self.pick_in_progress = True
            # 즉시 정지
            stop_msg = Twist()
            stop_msg.linear.x = 0.0
            stop_msg.angular.z = 0.0
            self.cmd_pub.publish(stop_msg)
            # pick 작업 수행
            if self.detected_marker_array is not None:
                self.pick_pub.publish(self.detected_marker_array)
                marker = self.detected_marker_array.markers[0]
                pos = marker.pose.pose.position
                ori = marker.pose.pose.orientation
                self.get_logger().info(
                    f":포장: Publishing pick_trigger with marker ID: {marker.id}\n"
                    f"    Position: x={pos.x:.2f}, y={pos.y:.2f}, z={pos.z:.2f}\n"
                    f"    Orientation: roll(x)={ori.x:.2f}, pitch(y)={ori.y:.2f}, yaw(z)={ori.z:.2f}"
                )
            # 상태 초기화
            self.marker_detected = False
            return  # 이후의 라인 추적 로직을 건너뜀
        # -----------------------
        # 아래는 평소 라인 추적 동작
        # -----------------------
        # decode
        frame = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # masks
        y_mask=cv2.inRange(hsv,(self.y_h[0],self.y_s[0],self.y_v[0]),(self.y_h[1],self.y_s[1],self.y_v[1]))
        w_mask=cv2.inRange(hsv,(self.w_h[0],self.w_s[0],self.w_v[0]),(self.w_h[1],self.w_s[1],self.w_v[1]))
        num_labels_white, _ = cv2.connectedComponents(w_mask)
        num_labels_yellow, _ = cv2.connectedComponents(y_mask)
        y_cnt, w_cnt = cv2.countNonZero(y_mask), cv2.countNonZero(w_mask)
        only_y = y_cnt>self.pixel_thresh and w_cnt<=self.pixel_thresh
        only_w = w_cnt>self.pixel_thresh and y_cnt<=self.pixel_thresh
        both   = y_cnt>self.pixel_thresh and w_cnt>self.pixel_thresh
        if only_y and self.state!='recovering':
            self.move_joint(+self.offset*1.1); self.state='recovering'; self.dir='right'; self.t0=time.time(); self.recover=True; self.cmd(0)
            return
        if only_w and self.state!='recovering':
            self.move_joint(-self.offset); self.state='recovering'; self.dir='left';  self.t0=time.time(); self.recover=True; self.cmd(0)
            return
        if self.recover:
            dt=time.time()-self.t0
            if dt<self.recover_wait: self.cmd(0); return
            ang = -0.2 if self.dir=='right' else +0.2
            self.cmd(0.05,ang)
            if both or dt>self.recover_spin:
                self.move_joint(0); self.state='restored'; self.recover=False
            return
        #PID 주행
        mask=cv2.bitwise_or(y_mask,w_mask)
        hist=np.sum(mask[h//2:,:],axis=0)
        mid=w//2; lx,rx=np.argmax(hist[:mid]),np.argmax(hist[mid:])+mid
        err=mid-((lx+rx)//2)
        self.integral = np.clip(self.integral+err, -self.i_max, self.i_max)
        deriv=err-self.prev_err; self.prev_err=err
        pid_ang=self.kp*err+self.ki*self.integral+self.kd*deriv
        ang=pid_ang+self.joints[0]; vel=self.max_v if abs(err)<50 else self.min_v
        self.cmd(vel,np.clip(ang,-self.max_a,self.max_a))
        # 디버그 퍼블리시
        dbg=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
        dbg[y_mask>0]=[0,255,255]; dbg[w_mask>0]=[255,255,255]
        self.debug_pub.publish(self.br.cv2_to_imgmsg(dbg,'bgr8'))
        """
        # PID 제어
        mask = cv2.bitwise_or(y_mask, w_mask)
        debug_frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        debug_frame[y_mask > 0] = [0, 255, 255]
        debug_frame[w_mask > 0] = [255, 255, 255]
        debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
        self.debug_frame_pub.publish(debug_msg)
        hist = np.sum(mask[h // 2:, :], axis=0)
        mid = w // 2
        lx = np.argmax(hist[:mid])
        rx = np.argmax(hist[mid:]) + mid
        lc = (lx + rx) // 2
        fc = mid
        err = fc - lc
        self.integral += err
        deriv = err - self.prev_error
        ang = self.kp * err + self.ki * self.integral + self.kd * deriv
        self.prev_error = err
        vel = self.max_v if abs(err) < 50 else self.min_v
        t = Twist()
        t.linear.x = vel
        t.angular.z = np.clip(ang, -self.max_a, self.max_a)
        self.cmd_pub.publish(t)
        """
def main():
    rclpy.init()
    node = LaneFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
if __name__=='__main__':
    main()

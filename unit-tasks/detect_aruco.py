import rclpy, cv2, time, numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, JointState, Image
from geometry_msgs.msg import Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class LaneFollower(Node):
    def __init__(self):
        super().__init__('lane_follower')
        # 파라미터 선언 & 읽기
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

        # 상태 초기화
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

    def loop(self):
        if self.frame is None: return
        h,w=self.frame.shape[:2]
        hsv=cv2.cvtColor(self.frame,cv2.COLOR_BGR2HSV)
        # 마스크
        y_mask=cv2.inRange(hsv,(self.y_h[0],self.y_s[0],self.y_v[0]),(self.y_h[1],self.y_s[1],self.y_v[1]))
        w_mask=cv2.inRange(hsv,(self.w_h[0],self.w_s[0],self.w_v[0]),(self.w_h[1],self.w_s[1],self.w_v[1]))
        y_cnt,w_cnt = cv2.countNonZero(y_mask), cv2.countNonZero(w_mask)
        only_y = y_cnt>self.pixel_thresh and w_cnt<=self.pixel_thresh
        only_w = w_cnt>self.pixel_thresh and y_cnt<=self.pixel_thresh
        both   = y_cnt>self.pixel_thresh and w_cnt>self.pixel_thresh

        # 복구 FSM
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

        # PID 주행
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

def main():
    rclpy.init(); node=LaneFollower(); rclpy.spin(node); node.destroy_node(); rclpy.shutdown()

if __name__=='__main__':
    main()

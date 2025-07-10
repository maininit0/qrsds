#!/usr/bin/env python3
import rclpy
import math
import time
import numpy as np
from rclpy.node import Node
from ai_msgs.msg import PerceptionTargets
from scservo_sdk import *

class AngleBasedHeadTiltServoController(Node):
    def __init__(self):
        super().__init__('angle_based_head_tilt_servo_controller')
        
        # 订阅人体识别话题
        self.subscription = self.create_subscription(
            PerceptionTargets,
            '/hobot_mono2d_body_detection',
            self.listener_callback,
            10)
        
        # 初始化舵机通信
        self.portHandler = PortHandler('/dev/ttyACM0')
        self.packetHandler = sms_sts(self.portHandler)
        
        # 舵机参数
        self.HEAD_SERVO_ID = 1  # 头部舵机
        self.ARM_SERVO_ID = 4   # 左臂舵机
        self.SPEED = 2750
        self.ACCEL = 250
        
        # 头部舵机角度定义
        self.TILT_RIGHT_ANGLE = 135   # 头右歪时舵机角度
        self.TILT_NEUTRAL_ANGLE = 90  # 头不歪时舵机角度
        self.TILT_LEFT_ANGLE = 45     # 头左歪时舵机角度
        
        # 左臂舵机角度定义
        self.ARM_DEFAULT_ANGLE = 180  # 未检测到人时的默认角度
        self.ARM_ANGLE_THRESHOLD = 5  # 角度变化阈值，超过此值才移动舵机
        
        # 头部倾斜判断阈值
        self.TILT_THRESHOLD = 0.15  # 偏移比例阈值（相对于肩膀宽度）
        self.MIN_SHOULDER_WIDTH = 30  # 最小肩膀宽度（像素），用于过滤无效检测
        
        # 超时检测参数
        self.last_detection_time = time.time()
        self.detection_timeout = 2.0
        
        # 当前舵机位置和状态
        self.current_head_angle = self.TILT_NEUTRAL_ANGLE
        self.current_arm_angle = self.ARM_DEFAULT_ANGLE
        self.last_real_arm_angle = 180  # 上一次检测到的真实手臂角度，初始化为180度
        
        # 初始化舵机连接
        self.init_servo()
        
        # 创建定时器
        self.timer = self.create_timer(1.0, self.check_detection_timeout)
        
        self.get_logger().info('基于人体姿态的舵机控制器已启动')

    def init_servo(self):
        """初始化舵机连接"""
        if not self.portHandler.openPort():
            self.get_logger().error("打开舵机端口失败")
            return False
        
        if not self.portHandler.setBaudRate(1000000):
            self.get_logger().error("设置波特率失败")
            self.portHandler.closePort()
            return False
        
        self.get_logger().info("舵机连接成功")
        self.move_servo(self.HEAD_SERVO_ID, self.TILT_NEUTRAL_ANGLE)
        self.move_servo(self.ARM_SERVO_ID, self.ARM_DEFAULT_ANGLE)
        return True

    def angle_to_position(self, angle):
        """将角度转换为舵机位置值"""
        angle = angle % 360
        return int((angle / 360) * 4095)

    def move_servo(self, servo_id, angle):
        """移动指定舵机到指定角度"""
        # 检查当前角度，避免重复移动
        if servo_id == self.HEAD_SERVO_ID and abs(angle - self.current_head_angle) < 1:
            return True
        elif servo_id == self.ARM_SERVO_ID and abs(angle - self.current_arm_angle) < 1:
            return True
            
        position = self.angle_to_position(angle)
        
        self.packetHandler.groupSyncWrite.clearParam()
        
        if self.packetHandler.SyncWritePosEx(servo_id, position, self.SPEED, self.ACCEL):
            result = self.packetHandler.groupSyncWrite.txPacket()
            self.packetHandler.groupSyncWrite.clearParam()
            
            if result == COMM_SUCCESS:
                if servo_id == self.HEAD_SERVO_ID:
                    self.current_head_angle = angle
                    self.get_logger().info(f"头部舵机移动到 {angle}°")
                elif servo_id == self.ARM_SERVO_ID:
                    self.current_arm_angle = angle
                    self.get_logger().info(f"左臂舵机移动到 {angle}°")
                return True
        
        self.get_logger().error(f"舵机{servo_id}控制失败")
        return False

    def calculate_angle_between_vectors(self, p1, p2, p3):
        """
        计算两个向量之间的夹角
        p2是顶点，计算从p2指向p1的向量与从p2指向p3的向量之间的夹角
        """
        # 向量1: p2 -> p1 (左肩到左肘)
        v1_x = p1.x - p2.x
        v1_y = p1.y - p2.y
        
        # 向量2: p2 -> p3 (左肩到左胯)
        v2_x = p3.x - p2.x
        v2_y = p3.y - p2.y
        
        # 计算向量的模长
        v1_length = math.sqrt(v1_x**2 + v1_y**2)
        v2_length = math.sqrt(v2_x**2 + v2_y**2)
        
        # 避免除零错误
        if v1_length == 0 or v2_length == 0:
            return 180  # 返回默认角度
        
        # 计算点积
        dot_product = v1_x * v2_x + v1_y * v2_y
        
        # 计算夹角的余弦值
        cos_angle = dot_product / (v1_length * v2_length)
        
        # 限制余弦值在[-1, 1]范围内，避免数值误差
        cos_angle = max(-1, min(1, cos_angle))
        
        # 计算夹角（弧度转角度）
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)
        
        return angle_deg

    def map_real_angle_to_servo_angle(self, real_angle):
        """
        将实际检测到的夹角映射到舵机角度
        根据示意图：实际180度 -> 舵机90度，实际90度 -> 舵机180度
        使用线性映射：servo_angle = 270 - real_angle
        """
        # 确保实际角度在合理范围内 (0-180度)
        real_angle = max(0, min(180, real_angle))
        
        # 根据示意图的线性关系映射
        # 当实际角度为180度时，舵机角度为90度
        # 当实际角度为90度时，舵机角度为180度
        # 线性关系：servo_angle = 270 - real_angle
        servo_angle = 270 - real_angle
        
        # 记录映射计算过程
        self.get_logger().info(f"角度映射: 实际角度 {real_angle:.1f}° -> 舵机角度 {servo_angle:.1f}° (使用公式: 270 - {real_angle:.1f})")
        
        # 舵机角度范围应该是 90-270度 (当实际角度为0-180度时)
        # 不再强制限制，让舵机按照计算出的角度工作
        return int(servo_angle)

    def calculate_left_arm_angle(self, keypoints, confidences):
        """
        计算左臂角度
        使用左肘、左肩、左胯三个点计算夹角
        左肩作为顶点，计算左肩-左肘连线与左肩-左胯连线的夹角
        keypoints[5]: 左肩
        keypoints[7]: 左肘
        keypoints[11]: 左胯
        """
        # 检查必要关键点的置信度
        left_shoulder_conf = confidences[5]   # 左肩
        left_elbow_conf = confidences[7]      # 左肘
        left_hip_conf = confidences[11]       # 左胯
        
        if left_shoulder_conf < 0.4 or left_elbow_conf < 0.4 or left_hip_conf < 0.4:
            self.get_logger().info("左臂关键点置信度不足，使用默认角度")
            return None
        
        # # 获取关键点坐标
        # left_shoulder = keypoints[5]  # 顶点
        # left_elbow = keypoints[7]     # 第一个点
        # left_hip = keypoints[11]      # 第二个点（参考垂直线）
        
        # # 计算左肩-左肘连线与左肩-左胯连线之间的夹角
        # # left_shoulder是顶点，left_elbow是第一个点，left_hip是第二个点
        # real_arm_angle = self.calculate_angle_between_vectors(left_elbow, left_shoulder, left_hip)
        
        # # 记录详细信息
        # self.get_logger().info(f"左肩位置: ({left_shoulder.x:.1f}, {left_shoulder.y:.1f}) - 顶点")
        # self.get_logger().info(f"左肘位置: ({left_elbow.x:.1f}, {left_elbow.y:.1f})")
        # self.get_logger().info(f"左胯位置: ({left_hip.x:.1f}, {left_hip.y:.1f}) - 参考垂直线")
        # self.get_logger().info(f"左臂实际夹角: {real_arm_angle:.1f}° (左肩-左肘线与左肩-左胯线之间)")

        # 获取关键点坐标
        left_shoulder = keypoints[5]  # 顶点
        left_elbow = keypoints[7]     # 第一个点
        
        # 创建一个简单的点类来表示左肩正下方的点
        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        
        # 创建左肩正下方的参考点（垂直向下50像素，距离可以调整）
        vertical_reference = Point(left_shoulder.x, left_shoulder.y + 50)
        
        # 计算左肩-左肘连线与左肩-垂直向下连线之间的夹角
        # left_shoulder是顶点，left_elbow是第一个点，vertical_reference是第二个点
        real_arm_angle = self.calculate_angle_between_vectors(left_elbow, left_shoulder, vertical_reference)
        
        # 记录详细信息
        self.get_logger().info(f"左肩位置: ({left_shoulder.x:.1f}, {left_shoulder.y:.1f}) - 顶点")
        self.get_logger().info(f"左肘位置: ({left_elbow.x:.1f}, {left_elbow.y:.1f})")
        self.get_logger().info(f"垂直参考点: ({vertical_reference.x:.1f}, {vertical_reference.y:.1f}) - 左肩正下方")
        self.get_logger().info(f"左臂实际夹角: {real_arm_angle:.1f}° (左肩-左肘线与垂直向下线之间)")  

        return real_arm_angle

    def determine_head_tilt_by_nose_shoulders(self, keypoints, confidences):
        """
        根据鼻子和左右肩膀的位置判断头部倾斜方向
        keypoints[0]: 鼻子
        keypoints[5]: 左肩
        keypoints[6]: 右肩
        """
        # 检查必要关键点的置信度
        nose_conf = confidences[0]      # 鼻子
        left_shoulder_conf = confidences[5]   # 左肩
        right_shoulder_conf = confidences[6]  # 右肩
        
        if nose_conf < 0.4 or left_shoulder_conf < 0.4 or right_shoulder_conf < 0.4:
            self.get_logger().info("关键点置信度不足，无法判断头部倾斜")
            return "neutral"
        
        # 获取关键点坐标
        nose = keypoints[0]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        
        # 计算肩膀中心点
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        
        # 计算肩膀宽度
        shoulder_width = abs(right_shoulder.x - left_shoulder.x)
        
        # 过滤肩膀宽度过小的情况（可能是检测错误）
        if shoulder_width < self.MIN_SHOULDER_WIDTH:
            self.get_logger().info(f"肩膀宽度过小: {shoulder_width:.1f}px，跳过判断")
            return "neutral"
        
        # 计算鼻子相对于肩膀中心的水平偏移
        nose_offset_x = nose.x - shoulder_center_x
        
        # 计算偏移比例（相对于肩膀宽度）
        offset_ratio = nose_offset_x / shoulder_width
        
        # 记录详细信息
        self.get_logger().info(f"鼻子位置: ({nose.x:.1f}, {nose.y:.1f})")
        self.get_logger().info(f"左肩位置: ({left_shoulder.x:.1f}, {left_shoulder.y:.1f})")
        self.get_logger().info(f"右肩位置: ({right_shoulder.x:.1f}, {right_shoulder.y:.1f})")
        self.get_logger().info(f"肩膀中心: ({shoulder_center_x:.1f}, {shoulder_center_y:.1f})")
        self.get_logger().info(f"肩膀宽度: {shoulder_width:.1f}px")
        self.get_logger().info(f"鼻子偏移: {nose_offset_x:.1f}px")
        self.get_logger().info(f"偏移比例: {offset_ratio:.3f}")
        
        # 根据偏移比例判断倾斜方向
        if offset_ratio > self.TILT_THRESHOLD:
            # 鼻子偏向右侧，说明头部向右倾斜
            self.get_logger().info(f"检测到右倾: 偏移比例 {offset_ratio:.3f} > 阈值 {self.TILT_THRESHOLD}")
            return "right"
        elif offset_ratio < -self.TILT_THRESHOLD:
            # 鼻子偏向左侧，说明头部向左倾斜
            self.get_logger().info(f"检测到左倾: 偏移比例 {offset_ratio:.3f} < 阈值 -{self.TILT_THRESHOLD}")
            return "left"
        else:
            # 鼻子在中间区域，头部保持垂直
            self.get_logger().info(f"头部垂直: 偏移比例 {offset_ratio:.3f} 在阈值范围内")
            return "neutral"

    def check_detection_timeout(self):
        """检查超时"""
        current_time = time.time()
        if current_time - self.last_detection_time > self.detection_timeout:
            if self.current_head_angle != self.TILT_NEUTRAL_ANGLE:
                self.get_logger().info("长时间未检测到人，头部回到默认位置")
                self.move_servo(self.HEAD_SERVO_ID, self.TILT_NEUTRAL_ANGLE)
            if self.current_arm_angle != self.ARM_DEFAULT_ANGLE:
                self.get_logger().info("长时间未检测到人，左臂回到默认位置(180°)")
                self.move_servo(self.ARM_SERVO_ID, self.ARM_DEFAULT_ANGLE)
                self.last_real_arm_angle = 180  # 重置为默认值

    def listener_callback(self, msg):
        """处理检测消息"""
        detected_person = False
        
        for target in msg.targets:
            if target.type == "person":
                for point_group in target.points:
                    if point_group.type == "body_kps" and len(point_group.point) >= 18:
                        keypoints = point_group.point
                        confidences = point_group.confidence
                        
                        # 检查必要的关键点
                        has_head_points = (
                            confidences[0] > 0.3 and    # 鼻子
                            confidences[5] > 0.3 and    # 左肩
                            confidences[6] > 0.3        # 右肩
                        )
                        
                        has_arm_points = (
                            confidences[5] > 0.3 and    # 左肩
                            confidences[7] > 0.3 and    # 左肘
                            confidences[11] > 0.3       # 左胯
                        )
                        
                        if has_head_points or has_arm_points:
                            detected_person = True
                            self.last_detection_time = time.time()
                            
                            # 控制头部舵机
                            if has_head_points:
                                tilt_direction = self.determine_head_tilt_by_nose_shoulders(keypoints, confidences)
                                
                                if tilt_direction == "left":
                                    self.get_logger().info("头部向左倾斜")
                                    self.move_servo(self.HEAD_SERVO_ID, self.TILT_LEFT_ANGLE)
                                elif tilt_direction == "right":
                                    self.get_logger().info("头部向右倾斜")
                                    self.move_servo(self.HEAD_SERVO_ID, self.TILT_RIGHT_ANGLE)
                                else:
                                    self.get_logger().info("头部保持垂直")
                                    self.move_servo(self.HEAD_SERVO_ID, self.TILT_NEUTRAL_ANGLE)
                            
                            # 控制左臂舵机
                            if has_arm_points:
                                real_arm_angle = self.calculate_left_arm_angle(keypoints, confidences)
                                
                                if real_arm_angle is not None:
                                    # 检查角度变化是否超过阈值
                                    angle_diff = abs(real_arm_angle - self.last_real_arm_angle)
                                    
                                    if angle_diff >= self.ARM_ANGLE_THRESHOLD:
                                        # 角度变化超过阈值，执行舵机控制
                                        servo_angle = self.map_real_angle_to_servo_angle(real_arm_angle)
                                        self.get_logger().info(f"左臂实际角度: {real_arm_angle:.1f}° -> 舵机角度: {servo_angle}°")
                                        self.get_logger().info(f"角度变化: {angle_diff:.1f}° >= {self.ARM_ANGLE_THRESHOLD}°，执行舵机控制")
                                        self.move_servo(self.ARM_SERVO_ID, servo_angle)
                                        self.last_real_arm_angle = real_arm_angle
                                    else:
                                        self.get_logger().info(f"角度变化不足: {angle_diff:.1f}° < {self.ARM_ANGLE_THRESHOLD}°，保持当前舵机位置")
                            
                            break
                
                if detected_person:
                    break
        
        if not detected_person:
            self.get_logger().debug("未检测到有效的人体关键点")

    def __del__(self):
        """析构函数"""
        try:
            self.move_servo(self.HEAD_SERVO_ID, self.TILT_NEUTRAL_ANGLE)
            self.move_servo(self.ARM_SERVO_ID, self.ARM_DEFAULT_ANGLE)
            self.portHandler.closePort()
            self.get_logger().info("端口已关闭")
        except:
            pass

def main(args=None):
    rclpy.init(args=args)
    controller = AngleBasedHeadTiltServoController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info("程序被用户中断")
    except Exception as e:
        controller.get_logger().error(f"发生错误: {str(e)}")
    finally:
        controller.move_servo(controller.HEAD_SERVO_ID, controller.TILT_NEUTRAL_ANGLE)
        controller.move_servo(controller.ARM_SERVO_ID, controller.ARM_DEFAULT_ANGLE)
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


#!/usr/bin/env python3
import rclpy
import math
import time
import numpy as np
from rclpy.node import Node
from ai_msgs.msg import PerceptionTargets
from scservo_sdk import *

class FullBodyServoController(Node):
    def __init__(self):
        super().__init__('full_body_servo_controller')
        
        # 订阅人体识别话题
        self.subscription = self.create_subscription(
            PerceptionTargets,
            '/hobot_mono2d_body_detection',
            self.listener_callback,
            10)
        
        # 初始化舵机通信
        self.portHandler = PortHandler('/dev/ttyACM0')
        self.packetHandler = sms_sts(self.portHandler)
        
        # 舵机ID定义
        self.HEAD_SERVO_ID = 1       # 头部舵机
        self.FOREARM_SERVO_ID = 2    # 左小臂舵机
        self.ARM_SERVO_ID = 4        # 左臂舵机
        self.THIGH_SERVO_ID = 6      # 左大腿舵机
        self.RIGHT_THIGH_SERVO_ID = 7  # 右大腿舵机
        self.LOWER_LEG_SERVO_ID = 8   # 左小腿舵机
        
        # 通用舵机参数
        self.SPEED = 2750
        self.ACCEL = 250
        
        # 头部舵机角度定义
        self.TILT_RIGHT_ANGLE = 135   # 头右歪时舵机角度
        self.TILT_NEUTRAL_ANGLE = 90  # 头不歪时舵机角度
        self.TILT_LEFT_ANGLE = 45     # 头左歪时舵机角度
        
        # 左小臂舵机角度定义
        self.FOREARM_DEFAULT_ANGLE = 180  # 未检测到人时的默认角度
        self.FOREARM_ANGLE_THRESHOLD = 5  # 角度变化阈值，超过此值才移动舵机
        
        # 左臂舵机角度定义
        self.ARM_DEFAULT_ANGLE = 180  # 未检测到人时的默认角度
        self.ARM_ANGLE_THRESHOLD = 5  # 角度变化阈值
        
        # 左大腿舵机角度定义
        self.THIGH_DEFAULT_ANGLE = 200  # 未检测到人时的默认角度
        self.THIGH_MIN_ANGLE = 110      # 舵机最小角度（对应实际0度）
        self.THIGH_MAX_ANGLE = 200      # 舵机最大角度（对应实际90度）
        self.THIGH_ANGLE_THRESHOLD = 5  # 角度变化阈值
        
        # 右大腿舵机角度定义
        self.RIGHT_THIGH_DEFAULT_ANGLE = 240  # 未检测到人时的默认角度
        self.RIGHT_THIGH_MIN_ANGLE = 240      # 舵机最小角度（对应实际90度）
        self.RIGHT_THIGH_MAX_ANGLE = 330      # 舵机最大角度（对应实际0度）
        self.RIGHT_THIGH_ANGLE_THRESHOLD = 5  # 角度变化阈值
        
        # 左小腿舵机角度定义
        self.LOWER_LEG_DEFAULT_ANGLE = 180  # 未检测到人时的默认角度
        self.LOWER_LEG_ANGLE_THRESHOLD = 5  # 角度变化阈值，超过此值才移动舵机
        
        # 头部倾斜判断阈值
        self.TILT_THRESHOLD = 0.15  # 偏移比例阈值（相对于肩膀宽度）
        self.MIN_SHOULDER_WIDTH = 30  # 最小肩膀宽度（像素），用于过滤无效检测
        
        # 超时检测参数
        self.last_detection_time = time.time()
        self.detection_timeout = 2.0
        
        # 当前舵机位置和状态
        self.current_head_angle = self.TILT_NEUTRAL_ANGLE
        self.current_forearm_angle = self.FOREARM_DEFAULT_ANGLE
        self.current_arm_angle = self.ARM_DEFAULT_ANGLE
        self.current_thigh_angle = self.THIGH_DEFAULT_ANGLE
        self.current_right_thigh_angle = self.RIGHT_THIGH_DEFAULT_ANGLE
        self.current_lower_leg_angle = self.LOWER_LEG_DEFAULT_ANGLE
        
        self.last_real_forearm_angle = 180  # 上一次检测到的真实小臂角度
        self.last_real_arm_angle = 180      # 上一次检测到的真实手臂角度
        self.last_real_thigh_angle = 90      # 上一次检测到的真实左大腿角度
        self.last_real_right_thigh_angle = 90  # 上一次检测到的真实右大腿角度
        self.last_real_lower_leg_angle = 180  # 上一次检测到的真实小腿角度
        
        # 初始化舵机连接
        self.init_servo()
        
        # 创建定时器
        self.timer = self.create_timer(1.0, self.check_detection_timeout)
        
        self.get_logger().info('全身舵机控制器已启动')

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
        # 初始化所有舵机到默认位置
        self.move_servo(self.HEAD_SERVO_ID, self.TILT_NEUTRAL_ANGLE)
        self.move_servo(self.FOREARM_SERVO_ID, self.FOREARM_DEFAULT_ANGLE)
        self.move_servo(self.ARM_SERVO_ID, self.ARM_DEFAULT_ANGLE)
        self.move_servo(self.THIGH_SERVO_ID, self.THIGH_DEFAULT_ANGLE)
        self.move_servo(self.RIGHT_THIGH_SERVO_ID, self.RIGHT_THIGH_DEFAULT_ANGLE)
        self.move_servo(self.LOWER_LEG_SERVO_ID, self.LOWER_LEG_DEFAULT_ANGLE)
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
        elif servo_id == self.FOREARM_SERVO_ID and abs(angle - self.current_forearm_angle) < 1:
            return True
        elif servo_id == self.ARM_SERVO_ID and abs(angle - self.current_arm_angle) < 1:
            return True
        elif servo_id == self.THIGH_SERVO_ID and abs(angle - self.current_thigh_angle) < 1:
            return True
        elif servo_id == self.RIGHT_THIGH_SERVO_ID and abs(angle - self.current_right_thigh_angle) < 1:
            return True
        elif servo_id == self.LOWER_LEG_SERVO_ID and abs(angle - self.current_lower_leg_angle) < 1:
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
                elif servo_id == self.FOREARM_SERVO_ID:
                    self.current_forearm_angle = angle
                    self.get_logger().info(f"左小臂舵机移动到 {angle}°")
                elif servo_id == self.ARM_SERVO_ID:
                    self.current_arm_angle = angle
                    self.get_logger().info(f"左臂舵机移动到 {angle}°")
                elif servo_id == self.THIGH_SERVO_ID:
                    self.current_thigh_angle = angle
                    self.get_logger().info(f"左大腿舵机移动到 {angle}°")
                elif servo_id == self.RIGHT_THIGH_SERVO_ID:
                    self.current_right_thigh_angle = angle
                    self.get_logger().info(f"右大腿舵机移动到 {angle}°")
                elif servo_id == self.LOWER_LEG_SERVO_ID:
                    self.current_lower_leg_angle = angle
                    self.get_logger().info(f"左小腿舵机移动到 {angle}°")
                return True
        
        self.get_logger().error(f"舵机{servo_id}控制失败")
        return False

    def calculate_angle_between_vectors(self, p1, p2, p3):
        """
        计算两个向量之间的夹角
        p2是顶点，计算从p2指向p1的向量与从p2指向p3的向量之间的夹角
        """
        # 向量1: p2 -> p1
        v1_x = p1.x - p2.x
        v1_y = p1.y - p2.y
        
        # 向量2: p2 -> p3
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

    def calculate_angle_with_horizontal(self, p1, p2):
        """
        计算两点连线与水平线的夹角
        p1: 起点（胯部）
        p2: 终点（膝盖）
        返回角度范围：0-90度
        """
        # 计算向量
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        
        # 避免除零错误
        if dx == 0:
            return 90.0  # 垂直线与水平线夹角为90度
        
        # 计算与水平线的夹角（弧度）
        angle_rad = math.atan(abs(dy) / abs(dx))
        
        # 转换为角度
        angle_deg = math.degrees(angle_rad)
        
        # 确保角度在0-90度范围内
        angle_deg = max(0, min(90, angle_deg))
        
        return angle_deg

    def calculate_forearm_angle(self, keypoints, confidences):
        """
        计算左小臂角度
        计算左肘指向左手腕和左肘指向左肩之间的夹角
        根据左肘指向左手腕的向量的垂直分量决定最终角度
        keypoints[5]: 左肩
        keypoints[7]: 左肘
        keypoints[9]: 左手腕
        """
        # 检查必要关键点的置信度
        left_shoulder_conf = confidences[5]   # 左肩
        left_elbow_conf = confidences[7]      # 左肘
        left_wrist_conf = confidences[9]      # 左手腕
        
        if left_shoulder_conf < 0.4 or left_elbow_conf < 0.4 or left_wrist_conf < 0.4:
            self.get_logger().info("左小臂关键点置信度不足，使用默认角度")
            return None
        
        # 获取关键点坐标
        left_shoulder = keypoints[5]  # 左肩
        left_elbow = keypoints[7]     # 左肘（顶点）
        left_wrist = keypoints[9]     # 左手腕
        
        # 计算左肘指向左手腕的向量
        elbow_to_wrist_x = left_wrist.x - left_elbow.x
        elbow_to_wrist_y = left_wrist.y - left_elbow.y
        
        # 计算左肘指向左肩的向量
        elbow_to_shoulder_x = left_shoulder.x - left_elbow.x
        elbow_to_shoulder_y = left_shoulder.y - left_elbow.y
        
        # 计算两个向量的模长
        wrist_vector_length = math.sqrt(elbow_to_wrist_x**2 + elbow_to_wrist_y**2)
        shoulder_vector_length = math.sqrt(elbow_to_shoulder_x**2 + elbow_to_shoulder_y**2)
        
        # 避免除零错误
        if wrist_vector_length == 0 or shoulder_vector_length == 0:
            self.get_logger().warning("向量长度为0，返回默认角度")
            return None
        
        # 计算点积
        dot_product = elbow_to_wrist_x * elbow_to_shoulder_x + elbow_to_wrist_y * elbow_to_shoulder_y
        
        # 计算夹角的余弦值
        cos_angle = dot_product / (wrist_vector_length * shoulder_vector_length)
        
        # 限制余弦值在[-1, 1]范围内，避免数值误差
        cos_angle = max(-1, min(1, cos_angle))
        
        # 计算夹角（弧度转角度）
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)
        
        # 判断左肘指向左手腕的向量的垂直分量
        # 如果y分量为负（向上），使用计算的夹角
        # 如果y分量为正（向下），使用360度减去夹角
        if elbow_to_wrist_y < 0:  # 向上分量（y轴向下为正）
            final_angle = angle_deg
            direction = "向上"
        else:  # 向下分量
            final_angle = 360 - angle_deg
            direction = "向下"
        
        # 记录详细信息
        self.get_logger().info(f"左肩位置: ({left_shoulder.x:.1f}, {left_shoulder.y:.1f})")
        self.get_logger().info(f"左肘位置: ({left_elbow.x:.1f}, {left_elbow.y:.1f}) - 顶点")
        self.get_logger().info(f"左手腕位置: ({left_wrist.x:.1f}, {left_wrist.y:.1f})")
        self.get_logger().info(f"肘->手腕向量: ({elbow_to_wrist_x:.1f}, {elbow_to_wrist_y:.1f})")
        self.get_logger().info(f"肘->肩膀向量: ({elbow_to_shoulder_x:.1f}, {elbow_to_shoulder_y:.1f})")
        self.get_logger().info(f"基础夹角: {angle_deg:.1f}°, 手腕方向: {direction}, 最终角度: {final_angle:.1f}°")
        
        return final_angle

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
        
        # 获取关键点坐标
        left_shoulder = keypoints[5]  # 顶点
        left_elbow = keypoints[7]     # 第一个点
        left_hip = keypoints[11]      # 第二个点（参考垂直线）
        
        # 计算左肩-左肘连线与左肩-左胯连线之间的夹角
        real_arm_angle = self.calculate_angle_between_vectors(left_elbow, left_shoulder, left_hip)
        
        # 记录详细信息
        self.get_logger().info(f"左肩位置: ({left_shoulder.x:.1f}, {left_shoulder.y:.1f}) - 顶点")
        self.get_logger().info(f"左肘位置: ({left_elbow.x:.1f}, {left_elbow.y:.1f})")
        self.get_logger().info(f"左胯位置: ({left_hip.x:.1f}, {left_hip.y:.1f}) - 参考垂直线")
        self.get_logger().info(f"左臂实际夹角: {real_arm_angle:.1f}° (左肩-左肘线与左肩-左胯线之间)")
        
        return real_arm_angle

    def calculate_left_thigh_angle(self, keypoints, confidences):
        """
        计算左大腿角度
        使用左胯和左膝两个点计算与水平线的夹角
        keypoints[11]: 左胯
        keypoints[13]: 左膝
        """
        # 检查必要关键点的置信度
        left_hip_conf = confidences[11]    # 左胯
        left_knee_conf = confidences[13]   # 左膝
        
        if left_hip_conf < 0.4 or left_knee_conf < 0.4:
            self.get_logger().info("左大腿关键点置信度不足，使用默认角度")
            return None
        
        # 获取关键点坐标
        left_hip = keypoints[11]   # 左胯
        left_knee = keypoints[13]  # 左膝
        
        # 计算左胯-左膝连线与水平线的夹角
        real_thigh_angle = self.calculate_angle_with_horizontal(left_hip, left_knee)
        
        # 记录详细信息
        self.get_logger().info(f"左胯位置: ({left_hip.x:.1f}, {left_hip.y:.1f})")
        self.get_logger().info(f"左膝位置: ({left_knee.x:.1f}, {left_knee.y:.1f})")
        self.get_logger().info(f"左大腿实际角度: {real_thigh_angle:.1f}° (左胯-左膝连线与水平线夹角)")
        
        return real_thigh_angle

    def calculate_right_thigh_angle(self, keypoints, confidences):
        """
        计算右大腿角度
        使用右胯和右膝两个点计算与水平线的夹角
        keypoints[12]: 右胯
        keypoints[14]: 右膝
        """
        # 检查必要关键点的置信度
        right_hip_conf = confidences[12]    # 右胯
        right_knee_conf = confidences[14]   # 右膝
        
        if right_hip_conf < 0.4 or right_knee_conf < 0.4:
            self.get_logger().info("右大腿关键点置信度不足，使用默认角度")
            return None
        
        # 获取关键点坐标
        right_hip = keypoints[12]   # 右胯
        right_knee = keypoints[14]  # 右膝
        
        # 计算右胯-右膝连线与水平线的夹角
        real_thigh_angle = self.calculate_angle_with_horizontal(right_hip, right_knee)
        
        # 记录详细信息
        self.get_logger().info(f"右胯位置: ({right_hip.x:.1f}, {right_hip.y:.1f})")
        self.get_logger().info(f"右膝位置: ({right_knee.x:.1f}, {right_knee.y:.1f})")
        self.get_logger().info(f"右大腿实际角度: {real_thigh_angle:.1f}° (右胯-右膝连线与水平线夹角)")
        
        return real_thigh_angle

    def calculate_lower_leg_angle(self, keypoints, confidences):
        """
        计算左小腿角度
        计算左膝指向左脚踝和左膝指向左股之间的夹角
        根据左膝指向左脚踝的向量的水平分量决定最终角度
        keypoints[13]: 左膝
        keypoints[11]: 左股(左胯)
        keypoints[15]: 左脚踝
        """
        # 检查必要关键点的置信度
        left_knee_conf = confidences[13]     # 左膝
        left_hip_conf = confidences[11]      # 左股(左胯)
        left_ankle_conf = confidences[15]    # 左脚踝
        
        if left_knee_conf < 0.4 or left_hip_conf < 0.4 or left_ankle_conf < 0.4:
            self.get_logger().info("左小腿关键点置信度不足，使用默认角度")
            return None
        
        # 获取关键点坐标
        left_knee = keypoints[13]   # 左膝（顶点）
        left_hip = keypoints[11]    # 左股
        left_ankle = keypoints[15]  # 左脚踝
        
        # 计算左膝指向左脚踝的向量
        knee_to_ankle_x = left_ankle.x - left_knee.x
        knee_to_ankle_y = left_ankle.y - left_knee.y
        
        # 计算左膝指向左股的向量
        knee_to_hip_x = left_hip.x - left_knee.x
        knee_to_hip_y = left_hip.y - left_knee.y
        
        # 计算两个向量的模长
        ankle_vector_length = math.sqrt(knee_to_ankle_x**2 + knee_to_ankle_y**2)
        hip_vector_length = math.sqrt(knee_to_hip_x**2 + knee_to_hip_y**2)
        
        # 避免除零错误
        if ankle_vector_length == 0 or hip_vector_length == 0:
            self.get_logger().warning("向量长度为0，返回默认角度")
            return None
        
        # 计算点积
        dot_product = knee_to_ankle_x * knee_to_hip_x + knee_to_ankle_y * knee_to_hip_y
        
        # 计算夹角的余弦值
        cos_angle = dot_product / (ankle_vector_length * hip_vector_length)
        
        # 限制余弦值在[-1, 1]范围内，避免数值误差
        cos_angle = max(-1, min(1, cos_angle))
        
        # 计算夹角（弧度转角度）
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)
        
        # 判断左膝指向左脚踝的向量的水平分量
        # 如果x分量为正（向右），使用计算的夹角
        # 如果x分量为负（向左），使用360度减去夹角
        if knee_to_ankle_x > 0:  # 向右分量
            final_angle = angle_deg
            direction = "向右"
        else:  # 向左分量
            final_angle = 360 - angle_deg
            direction = "向左"
        
        # 记录详细信息
        self.get_logger().info(f"左膝位置: ({left_knee.x:.1f}, {left_knee.y:.1f}) - 顶点")
        self.get_logger().info(f"左股位置: ({left_hip.x:.1f}, {left_hip.y:.1f})")
        self.get_logger().info(f"左脚踝位置: ({left_ankle.x:.1f}, {left_ankle.y:.1f})")
        self.get_logger().info(f"膝->脚踝向量: ({knee_to_ankle_x:.1f}, {knee_to_ankle_y:.1f})")
        self.get_logger().info(f"膝->股向量: ({knee_to_hip_x:.1f}, {knee_to_hip_y:.1f})")
        self.get_logger().info(f"基础夹角: {angle_deg:.1f}°, 脚踝方向: {direction}, 最终角度: {final_angle:.1f}°")
        
        return final_angle

    def map_real_angle_to_servo_angle(self, real_angle):
        """
        将实际检测到的左臂夹角映射到舵机角度
        根据示意图：实际180度 -> 舵机90度，实际90度 -> 舵机180度
        使用线性映射：servo_angle = 270 - real_angle
        """
        # 确保实际角度在合理范围内 (0-180度)
        real_angle = max(0, min(180, real_angle))
        
        # 根据示意图的线性关系映射
        servo_angle = 270 - real_angle
        
        # 记录映射计算过程
        self.get_logger().info(f"角度映射: 实际角度 {real_angle:.1f}° -> 舵机角度 {servo_angle:.1f}° (使用公式: 270 - {real_angle:.1f})")
        
        return int(servo_angle)

    def map_thigh_angle_to_servo_angle(self, real_angle):
        """
        将实际检测到的左大腿角度映射到舵机角度
        实际90度 -> 舵机200度
        实际0度 -> 舵机110度
        使用线性映射
        """
        # 确保实际角度在合理范围内 (0-90度)
        real_angle = max(0, min(90, real_angle))
        
        # 线性映射：servo_angle = 110 + (real_angle / 90) * (200 - 110)
        servo_angle = 110 + (real_angle / 90) * (200 - 110)
        servo_angle = int(servo_angle)
        
        # 确保舵机角度在允许范围内
        servo_angle = max(self.THIGH_MIN_ANGLE, min(self.THIGH_MAX_ANGLE, servo_angle))
        
        # 记录映射计算过程
        self.get_logger().info(f"左大腿角度映射: 实际角度 {real_angle:.1f}° -> 舵机角度 {servo_angle}° (范围: {self.THIGH_MIN_ANGLE}-{self.THIGH_MAX_ANGLE})")
        
        return servo_angle

    def map_right_thigh_angle_to_servo_angle(self, real_angle):
        """
        将实际检测到的右大腿角度映射到舵机角度
        实际90度 -> 舵机240度
        实际0度 -> 舵机330度
        使用线性映射
        """
        # 确保实际角度在合理范围内 (0-90度)
        real_angle = max(0, min(90, real_angle))
        
        # 线性映射：servo_angle = 330 - (real_angle / 90) * (330 - 240)
        servo_angle = 330 - (real_angle / 90) * (330 - 240)
        servo_angle = int(servo_angle)
        
        # 确保舵机角度在允许范围内
        servo_angle = max(self.RIGHT_THIGH_MIN_ANGLE, min(self.RIGHT_THIGH_MAX_ANGLE, servo_angle))
        
        # 记录映射计算过程
        self.get_logger().info(f"右大腿角度映射: 实际角度 {real_angle:.1f}° -> 舵机角度 {servo_angle}° (范围: {self.RIGHT_THIGH_MIN_ANGLE}-{self.RIGHT_THIGH_MAX_ANGLE})")
        
        return servo_angle

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
            if self.current_forearm_angle != self.FOREARM_DEFAULT_ANGLE:
                self.get_logger().info("长时间未检测到人，左小臂回到默认位置(180°)")
                self.move_servo(self.FOREARM_SERVO_ID, self.FOREARM_DEFAULT_ANGLE)
                self.last_real_forearm_angle = 180  # 重置为默认值
            if self.current_arm_angle != self.ARM_DEFAULT_ANGLE:
                self.get_logger().info("长时间未检测到人，左臂回到默认位置(180°)")
                self.move_servo(self.ARM_SERVO_ID, self.ARM_DEFAULT_ANGLE)
                self.last_real_arm_angle = 180  # 重置为默认值
            if self.current_thigh_angle != self.THIGH_DEFAULT_ANGLE:
                self.get_logger().info("长时间未检测到人，左大腿回到默认位置(200°)")
                self.move_servo(self.THIGH_SERVO_ID, self.THIGH_DEFAULT_ANGLE)
                self.last_real_thigh_angle = 90  # 重置为默认值
            if self.current_right_thigh_angle != self.RIGHT_THIGH_DEFAULT_ANGLE:
                self.get_logger().info("长时间未检测到人，右大腿回到默认位置(240°)")
                self.move_servo(self.RIGHT_THIGH_SERVO_ID, self.RIGHT_THIGH_DEFAULT_ANGLE)
                self.last_real_right_thigh_angle = 90  # 重置为默认值
            if self.current_lower_leg_angle != self.LOWER_LEG_DEFAULT_ANGLE:
                self.get_logger().info("长时间未检测到人，左小腿回到默认位置(180°)")
                self.move_servo(self.LOWER_LEG_SERVO_ID, self.LOWER_LEG_DEFAULT_ANGLE)
                self.last_real_lower_leg_angle = 180  # 重置为默认值

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
                        
                        has_forearm_points = (
                            confidences[5] > 0.3 and    # 左肩
                            confidences[7] > 0.3 and    # 左肘
                            confidences[9] > 0.3        # 左手腕
                        )
                        
                        has_arm_points = (
                            confidences[5] > 0.3 and    # 左肩
                            confidences[7] > 0.3 and    # 左肘
                            confidences[11] > 0.3       # 左胯
                        )
                        
                        has_left_thigh_points = (
                            confidences[11] > 0.3 and   # 左胯
                            confidences[13] > 0.3       # 左膝
                        )
                        
                        has_right_thigh_points = (
                            confidences[12] > 0.3 and   # 右胯
                            confidences[14] > 0.3       # 右膝
                        )
                        
                        has_lower_leg_points = (
                            confidences[13] > 0.3 and   # 左膝
                            confidences[11] > 0.3 and   # 左股(左胯)
                            confidences[15] > 0.3       # 左脚踝
                        )
                        
                        if (has_head_points or has_forearm_points or has_arm_points or 
                            has_left_thigh_points or has_right_thigh_points or has_lower_leg_points):
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
                            
                            # 控制左小臂舵机
                            if has_forearm_points:
                                real_forearm_angle = self.calculate_forearm_angle(keypoints, confidences)
                                
                                if real_forearm_angle is not None:
                                    # 检查角度变化是否超过阈值
                                    angle_diff = abs(real_forearm_angle - self.last_real_forearm_angle)
                                    
                                    if angle_diff >= self.FOREARM_ANGLE_THRESHOLD:
                                        # 角度变化超过阈值，执行舵机控制
                                        servo_angle = int(real_forearm_angle)
                                        self.get_logger().info(f"左小臂实际角度: {real_forearm_angle:.1f}° -> 舵机角度: {servo_angle}°")
                                        self.get_logger().info(f"角度变化: {angle_diff:.1f}° >= {self.FOREARM_ANGLE_THRESHOLD}°，执行舵机控制")
                                        self.move_servo(self.FOREARM_SERVO_ID, servo_angle)
                                        self.last_real_forearm_angle = real_forearm_angle
                                    else:
                                        self.get_logger().info(f"角度变化不足: {angle_diff:.1f}° < {self.FOREARM_ANGLE_THRESHOLD}°，保持当前舵机位置")
                            
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
                            
                            # 控制左大腿舵机
                            if has_left_thigh_points:
                                real_thigh_angle = self.calculate_left_thigh_angle(keypoints, confidences)
                                
                                if real_thigh_angle is not None:
                                    # 检查角度变化是否超过阈值
                                    angle_diff = abs(real_thigh_angle - self.last_real_thigh_angle)
                                    
                                    if angle_diff >= self.THIGH_ANGLE_THRESHOLD:
                                        # 角度变化超过阈值，执行舵机控制
                                        servo_angle = self.map_thigh_angle_to_servo_angle(real_thigh_angle)
                                        self.get_logger().info(f"左大腿实际角度: {real_thigh_angle:.1f}° -> 舵机角度: {servo_angle}°")
                                        self.get_logger().info(f"角度变化: {angle_diff:.1f}° >= {self.THIGH_ANGLE_THRESHOLD}°，执行舵机控制")
                                        self.move_servo(self.THIGH_SERVO_ID, servo_angle)
                                        self.last_real_thigh_angle = real_thigh_angle
                                    else:
                                        self.get_logger().info(f"左大腿角度变化不足: {angle_diff:.1f}° < {self.THIGH_ANGLE_THRESHOLD}°，保持当前舵机位置")
                            
                            # 控制右大腿舵机
                            if has_right_thigh_points:
                                real_right_thigh_angle = self.calculate_right_thigh_angle(keypoints, confidences)
                                
                                if real_right_thigh_angle is not None:
                                    # 检查角度变化是否超过阈值
                                    angle_diff = abs(real_right_thigh_angle - self.last_real_right_thigh_angle)
                                    
                                    if angle_diff >= self.RIGHT_THIGH_ANGLE_THRESHOLD:
                                        # 角度变化超过阈值，执行舵机控制
                                        servo_angle = self.map_right_thigh_angle_to_servo_angle(real_right_thigh_angle)
                                        self.get_logger().info(f"右大腿实际角度: {real_right_thigh_angle:.1f}° -> 舵机角度: {servo_angle}°")
                                        self.get_logger().info(f"角度变化: {angle_diff:.1f}° >= {self.RIGHT_THIGH_ANGLE_THRESHOLD}°，执行舵机控制")
                                        self.move_servo(self.RIGHT_THIGH_SERVO_ID, servo_angle)
                                        self.last_real_right_thigh_angle = real_right_thigh_angle
                                    else:
                                        self.get_logger().info(f"右大腿角度变化不足: {angle_diff:.1f}° < {self.RIGHT_THIGH_ANGLE_THRESHOLD}°，保持当前舵机位置")
                            
                            # 控制左小腿舵机
                            if has_lower_leg_points:
                                real_lower_leg_angle = self.calculate_lower_leg_angle(keypoints, confidences)
                                
                                if real_lower_leg_angle is not None:
                                    # 检查角度变化是否超过阈值
                                    angle_diff = abs(real_lower_leg_angle - self.last_real_lower_leg_angle)
                                    
                                    if angle_diff >= self.LOWER_LEG_ANGLE_THRESHOLD:
                                        # 角度变化超过阈值，执行舵机控制
                                        servo_angle = int(real_lower_leg_angle)
                                        self.get_logger().info(f"左小腿实际角度: {real_lower_leg_angle:.1f}° -> 舵机角度: {servo_angle}°")
                                        self.get_logger().info(f"角度变化: {angle_diff:.1f}° >= {self.LOWER_LEG_ANGLE_THRESHOLD}°，执行舵机控制")
                                        self.move_servo(self.LOWER_LEG_SERVO_ID, servo_angle)
                                        self.last_real_lower_leg_angle = real_lower_leg_angle
                                    else:
                                        self.get_logger().info(f"角度变化不足: {angle_diff:.1f}° < {self.LOWER_LEG_ANGLE_THRESHOLD}°，保持当前舵机位置")
                            
                            break
                
                if detected_person:
                    break
        
        if not detected_person:
            self.get_logger().debug("未检测到有效的人体关键点")

    def __del__(self):
        """析构函数"""
        try:
            # 关闭所有舵机
            self.move_servo(self.HEAD_SERVO_ID, self.TILT_NEUTRAL_ANGLE)
            self.move_servo(self.FOREARM_SERVO_ID, self.FOREARM_DEFAULT_ANGLE)
            self.move_servo(self.ARM_SERVO_ID, self.ARM_DEFAULT_ANGLE)
            self.move_servo(self.THIGH_SERVO_ID, self.THIGH_DEFAULT_ANGLE)
            self.move_servo(self.RIGHT_THIGH_SERVO_ID, self.RIGHT_THIGH_DEFAULT_ANGLE)
            self.move_servo(self.LOWER_LEG_SERVO_ID, self.LOWER_LEG_DEFAULT_ANGLE)
            self.portHandler.closePort()
            self.get_logger().info("端口已关闭")
        except:
            pass

def main(args=None):
    rclpy.init(args=args)
    controller = FullBodyServoController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info("程序被用户中断")
    except Exception as e:
        controller.get_logger().error(f"发生错误: {str(e)}")
    finally:
        # 确保所有舵机回到默认位置
        controller.move_servo(controller.HEAD_SERVO_ID, controller.TILT_NEUTRAL_ANGLE)
        controller.move_servo(controller.FOREARM_SERVO_ID, controller.FOREARM_DEFAULT_ANGLE)
        controller.move_servo(controller.ARM_SERVO_ID, controller.ARM_DEFAULT_ANGLE)
        controller.move_servo(controller.THIGH_SERVO_ID, controller.THIGH_DEFAULT_ANGLE)
        controller.move_servo(controller.RIGHT_THIGH_SERVO_ID, controller.RIGHT_THIGH_DEFAULT_ANGLE)
        controller.move_servo(controller.LOWER_LEG_SERVO_ID, controller.LOWER_LEG_DEFAULT_ANGLE)
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
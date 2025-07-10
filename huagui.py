#!/usr/bin/env python3
import rclpy
import math
import time
import serial
import numpy as np
from rclpy.node import Node
from ai_msgs.msg import PerceptionTargets

class HumanTrackingSlider(Node):
    def __init__(self):
        super().__init__('human_tracking_slider')
        
        # 订阅人体识别话题
        self.subscription = self.create_subscription(
            PerceptionTargets,
            '/hobot_mono2d_body_detection',
            self.listener_callback,
            10)
        
        # 初始化串口通信
        try:
            self.serial_port = serial.Serial(
                port='/dev/ttyACM1',
                baudrate=115200,
                timeout=1.0
            )
            self.get_logger().info("串口连接成功: /dev/ttyACM1")
        except Exception as e:
            self.get_logger().error(f"串口连接失败: {str(e)}")
            self.serial_port = None
        
        # 映射参数
        self.TRACK_MIN_X = 220        # 跟踪区域最小X坐标
        self.TRACK_MAX_X = 440        # 跟踪区域最大X坐标
        self.SLIDER_MIN = 0           # 滑轨最小位置
        self.SLIDER_MAX = 12          # 滑轨最大位置
        self.NO_PERSON_POSITION = 0   # 无人位置
        
        self.current_slider_pos = self.NO_PERSON_POSITION  # 初始位置为0
        
        # 位置平滑参数
        self.position_history = []
        self.HISTORY_SIZE = 8         # 历史记录大小
        
        # 超时检测参数
        self.last_detection_time = time.time()
        self.detection_timeout = 2.0
        
        # 创建定时器
        self.timer = self.create_timer(1.0, self.check_detection_timeout)
        
        # 发送初始位置
        self.send_slider_position(self.NO_PERSON_POSITION)
        
        self.get_logger().info('人体追踪滑轨控制器已启动')
        self.get_logger().info(f'跟踪区域: X坐标{self.TRACK_MIN_X}-{self.TRACK_MAX_X}')
        self.get_logger().info(f'滑轨位置: {self.SLIDER_MIN}-{self.SLIDER_MAX}')
        self.get_logger().info(f'映射关系: X={self.TRACK_MIN_X}→位置{self.SLIDER_MIN}, X={self.TRACK_MAX_X}→位置{self.SLIDER_MAX}')
        self.get_logger().info(f'无人检测: 发送位置 {self.NO_PERSON_POSITION}')

    def map_position(self, x_coord):
        """将画面X坐标(220-440)映射到滑轨位置(0-12)"""
        # 如果人体在跟踪区域外，返回None表示不跟踪
        if x_coord < self.TRACK_MIN_X or x_coord > self.TRACK_MAX_X:
            return None
        
        # 线性映射: (x_coord - TRACK_MIN_X) / (TRACK_MAX_X - TRACK_MIN_X) * (SLIDER_MAX - SLIDER_MIN) + SLIDER_MIN
        track_range = self.TRACK_MAX_X - self.TRACK_MIN_X
        slider_range = self.SLIDER_MAX - self.SLIDER_MIN
        
        normalized_x = (x_coord - self.TRACK_MIN_X) / track_range
        slider_pos = normalized_x * slider_range + self.SLIDER_MIN
        
        # 四舍五入到整数
        slider_pos = round(slider_pos)
        
        # 确保在滑轨范围内
        slider_pos = max(self.SLIDER_MIN, min(slider_pos, self.SLIDER_MAX))
        
        return slider_pos

    def smooth_position(self, new_position):
        """平滑位置，减少抖动"""
        # 添加新位置到历史记录
        self.position_history.append(new_position)
        
        # 保持历史记录长度不超过HISTORY_SIZE
        if len(self.position_history) > self.HISTORY_SIZE:
            self.position_history.pop(0)
        
        # 计算加权平均，最新的位置权重更大
        if len(self.position_history) == 1:
            return new_position
        
        # 使用加权平均进行平滑
        weights = np.linspace(0.5, 1.0, len(self.position_history))
        weighted_sum = sum(pos * weight for pos, weight in zip(self.position_history, weights))
        weight_sum = sum(weights)
        
        smooth_pos = weighted_sum / weight_sum
        
        # 四舍五入到整数
        return round(smooth_pos)

    def send_slider_position(self, position):
        """发送位置指令到单片机"""
        if self.serial_port is None or not self.serial_port.is_open:
            self.get_logger().error("串口未连接，无法发送位置指令")
            return False
        
        try:
            # 格式化指令: (位置,0)
            command = f"({position},0)\n"
            self.serial_port.write(command.encode('utf-8'))
            self.current_slider_pos = position
            self.get_logger().info(f"发送滑轨位置: {position}")
            return True
        except Exception as e:
            self.get_logger().error(f"发送串口指令失败: {str(e)}")
            return False

    def check_detection_timeout(self):
        """检查超时"""
        current_time = time.time()
        if current_time - self.last_detection_time > self.detection_timeout:
            # 长时间未检测到人，发送位置0
            if self.current_slider_pos != self.NO_PERSON_POSITION:
                self.get_logger().info(f"长时间未检测到人，发送位置{self.NO_PERSON_POSITION}")
                self.send_slider_position(self.NO_PERSON_POSITION)
                # 清空历史记录
                self.position_history = []

    def listener_callback(self, msg):
        """处理检测消息"""
        detected_person = False
        person_in_track_area = False
        
        for target in msg.targets:
            if target.type == "person":
                for point_group in target.points:
                    if point_group.type == "body_kps" and len(point_group.point) >= 17:
                        keypoints = point_group.point
                        confidences = point_group.confidence
                        
                        # 检查必要的关键点 (左股和右股)
                        left_hip_conf = confidences[11]   # 左股
                        right_hip_conf = confidences[12]  # 右股
                        
                        if left_hip_conf > 0.3 and right_hip_conf > 0.3:
                            detected_person = True
                            self.last_detection_time = time.time()
                            
                            # 获取左股和右股的位置
                            left_hip = keypoints[11]
                            right_hip = keypoints[12]
                            
                            # 计算人体中心位置 (左股和右股的中点)
                            human_center_x = (left_hip.x + right_hip.x) / 2
                            
                            # 记录详细信息
                            self.get_logger().info(f"左股位置: ({left_hip.x:.1f}, {left_hip.y:.1f})")
                            self.get_logger().info(f"右股位置: ({right_hip.x:.1f}, {right_hip.y:.1f})")
                            self.get_logger().info(f"人体中心X坐标: {human_center_x:.1f}")
                            
                            # 将画面坐标映射到滑轨位置
                            mapped_position = self.map_position(human_center_x)
                            
                            if mapped_position is not None:
                                person_in_track_area = True
                                
                                # 平滑处理位置
                                smooth_pos = self.smooth_position(mapped_position)
                                
                                self.get_logger().info(f"在跟踪区域内，映射位置: {mapped_position}, 平滑后位置: {smooth_pos}")
                                
                                # 如果位置有变化，则发送指令
                                if smooth_pos != self.current_slider_pos:
                                    self.send_slider_position(smooth_pos)
                            else:
                                self.get_logger().info(f"人体在跟踪区域外 (X={human_center_x:.1f})")
                            
                            break
                
                if detected_person:
                    break
        
        # 如果没有检测到人，或者人不在跟踪区域内，发送位置0
        if not detected_person or not person_in_track_area:
            if self.current_slider_pos != self.NO_PERSON_POSITION:
                if not detected_person:
                    self.get_logger().info(f"未检测到人，发送位置{self.NO_PERSON_POSITION}")
                else:
                    self.get_logger().info(f"人体不在跟踪区域内，发送位置{self.NO_PERSON_POSITION}")
                self.send_slider_position(self.NO_PERSON_POSITION)
                # 清空历史记录
                self.position_history = []

    def __del__(self):
        """析构函数"""
        try:
            # 关闭串口
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()
                self.get_logger().info("串口已关闭")
        except:
            pass

def main(args=None):
    rclpy.init(args=args)
    slider_controller = HumanTrackingSlider()
    
    try:
        rclpy.spin(slider_controller)
    except KeyboardInterrupt:
        slider_controller.get_logger().info("程序被用户中断")
    except Exception as e:
        slider_controller.get_logger().error(f"发生错误: {str(e)}")
    finally:
        slider_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

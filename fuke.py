#!/usr/bin/env python3
import rclpy
import json
import time
from rclpy.node import Node
from scservo_sdk import *

class MotionReplayer(Node):
    def __init__(self):
        super().__init__('motion_replayer')
        
        # 初始化舵机通信
        self.portHandler = PortHandler('/dev/ttyACM0')
        self.packetHandler = sms_sts(self.portHandler)
        
        # 舵机ID定义（与记录时一致）
        self.HEAD_SERVO_ID = 1       # 头部舵机
        self.FOREARM_SERVO_ID = 2    # 左小臂舵机
        self.RIGHT_FOREARM_SERVO_ID = 3  # 右小臂舵机
        self.ARM_SERVO_ID = 4        # 左臂舵机
        self.RIGHT_ARM_SERVO_ID = 5   # 右臂舵机
        self.THIGH_SERVO_ID = 6      # 左大腿舵机
        self.RIGHT_THIGH_SERVO_ID = 7  # 右大腿舵机
        self.LOWER_LEG_SERVO_ID = 8   # 左小腿舵机
        self.RIGHT_LOWER_LEG_SERVO_ID = 9  # 右小腿舵机
        
        # 通用舵机参数
        self.SPEED = 2750
        self.ACCEL = 250
        
        # 动作数据
        self.motion_data = []
        self.current_frame = 0
        self.start_time = 0
        self.is_playing = False
        
        # 初始化舵机连接
        if not self.init_servo():
            self.get_logger().error("舵机初始化失败，程序退出")
            return
        
        # 创建定时器
        self.timer = self.create_timer(0.02, self.playback_loop)  # 50Hz更新频率
        self.get_logger().info('动作复刻程序已启动')

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
        return True

    def load_motion_data(self, file_path):
        """加载动作数据文件"""
        try:
            with open(file_path, 'r') as f:
                self.motion_data = json.load(f)
            self.get_logger().info(f"成功加载动作文件: {file_path}, 共 {len(self.motion_data)} 帧数据")
            return True
        except Exception as e:
            self.get_logger().error(f"加载动作文件失败: {str(e)}")
            return False

    def start_playback(self):
        """开始播放动作"""
        if not self.motion_data:
            self.get_logger().error("没有可播放的动作数据")
            return False
        
        self.current_frame = 0
        self.start_time = time.time()
        self.is_playing = True
        self.get_logger().info("开始播放动作...")
        return True

    def stop_playback(self):
        """停止播放动作"""
        self.is_playing = False
        self.get_logger().info("动作播放停止")

    def angle_to_position(self, angle):
        """将角度转换为舵机位置值"""
        angle = angle % 360
        return int((angle / 360) * 4095)

    def move_servo(self, servo_id, angle):
        """移动指定舵机到指定角度"""
        position = self.angle_to_position(angle)
        
        self.packetHandler.groupSyncWrite.clearParam()
        
        if self.packetHandler.SyncWritePosEx(servo_id, position, self.SPEED, self.ACCEL):
            result = self.packetHandler.groupSyncWrite.txPacket()
            self.packetHandler.groupSyncWrite.clearParam()
            return result == COMM_SUCCESS
        
        return False

    def playback_loop(self):
        """播放循环"""
        if not self.is_playing or not self.motion_data:
            return
        
        current_time = time.time() - self.start_time
        
        # 检查是否已经播放完所有帧
        if self.current_frame >= len(self.motion_data):
            self.stop_playback()
            return
        
        # 获取当前帧数据
        frame = self.motion_data[self.current_frame]
        
        # 检查是否到达该帧的时间点
        if current_time >= frame['time']:
            # 控制所有舵机
            self.move_servo(self.HEAD_SERVO_ID, frame['head'])
            self.move_servo(self.FOREARM_SERVO_ID, frame['left_forearm'])
            self.move_servo(self.RIGHT_FOREARM_SERVO_ID, frame['right_forearm'])
            self.move_servo(self.ARM_SERVO_ID, frame['left_arm'])
            self.move_servo(self.RIGHT_ARM_SERVO_ID, frame['right_arm'])
            self.move_servo(self.THIGH_SERVO_ID, frame['left_thigh'])
            self.move_servo(self.RIGHT_THIGH_SERVO_ID, frame['right_thigh'])
            self.move_servo(self.LOWER_LEG_SERVO_ID, frame['left_lower_leg'])
            self.move_servo(self.RIGHT_LOWER_LEG_SERVO_ID, frame['right_lower_leg'])
            
            # 输出当前帧信息
            self.get_logger().info(
                f"播放帧 {self.current_frame+1}/{len(self.motion_data)} "
                f"时间: {frame['time']:.2f}s "
                f"角度: 头{frame['head']}° "
                f"左臂[{frame['left_arm']}°,{frame['left_forearm']}°] "
                f"右臂[{frame['right_arm']}°,{frame['right_forearm']}°] "
                f"左腿[{frame['left_thigh']}°,{frame['left_lower_leg']}°] "
                f"右腿[{frame['right_thigh']}°,{frame['right_lower_leg']}°]"
            )
            
            self.current_frame += 1

    def __del__(self):
        """析构函数"""
        try:
            self.stop_playback()
            self.portHandler.closePort()
            self.get_logger().info("端口已关闭")
        except:
            pass

def main(args=None):
    rclpy.init(args=args)
    replayer = MotionReplayer()
    
    # 加载动作文件（替换为你的JSON文件路径）
    motion_file = "servo_angles_20250710_052420.json"  # 修改为你的文件名
    if not replayer.load_motion_data(motion_file):
        replayer.destroy_node()
        rclpy.shutdown()
        return
    
    # 开始播放
    replayer.start_playback()
    
    try:
        while rclpy.ok() and replayer.is_playing:
            rclpy.spin_once(replayer)
    except KeyboardInterrupt:
        replayer.get_logger().info("程序被用户中断")
    except Exception as e:
        replayer.get_logger().error(f"发生错误: {str(e)}")
    finally:
        replayer.stop_playback()
        replayer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
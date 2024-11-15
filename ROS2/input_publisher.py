# input_publisher.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import time

class InputPublisher(Node):
    def __init__(self):
        super().__init__('input_publisher')
        self.input_publishers = [
            self.create_publisher(Float32MultiArray, f'input_data_{i}', 10) for i in range(4)
        ]
        self.counter = 0  # 순차적으로 각 토픽에 전송하기 위한 카운터
        self.time_counter = 0.0
        # 타이머로 주기적으로 입력 데이터 퍼블리시
        self.timer = self.create_timer(0.5, self.publish_input_data)

    def publish_input_data(self):
        
        self.msg = Float32MultiArray()
        self.time_counter += 1.0
        if self.time_counter > 1000.0 :
            self.time_counter = 0.0
        current_time = [self.time_counter]

        # 임의의 입력 데이터 생성
        input_data = current_time + np.random.rand(1, 10).flatten().tolist()

        # 메시지에 타임스탬프와 데이터 포함
        self.msg.data = input_data
        
        # 순차적으로 각 모델에 데이터 전송
        self.input_publishers[self.counter].publish(self.msg)
        self.get_logger().info(f'Published input data to model {self.counter} at {self.msg.data[0]}')

        # 다음 모델로 카운터 이동
        self.counter = (self.counter + 1) % 4

def main():
    rclpy.init()  # ROS 2 런타임 초기화
    node = InputPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()  # ROS 2 런타임 종료
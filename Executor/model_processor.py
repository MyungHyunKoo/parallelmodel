import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
import torch
import torch.nn as nn
import numpy as np
import time
import asyncio
import threading

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class ModelProcessor(Node):
    def __init__(self, model_id):
        super().__init__(f'model_processor')
        # ReentrantCallbackGroup 생성
        self.callback_group = ReentrantCallbackGroup()

        # 구독자를 저장할 리스트
        self.list_subscriptions = []

        # model_id 값만큼 구독자 생성

        subscription = self.create_subscription(
            Float32MultiArray,
            f'input_data_0',  # 각 모델의 input_data 토픽
            self.listener_callback,  # 공통 콜백 함수 사용
            10,
            callback_group=self.callback_group
        )
        self.list_subscriptions.append(subscription)  # 구독자 리스트에 저장
        
        subscription = self.create_subscription(
            Float32MultiArray,
            f'input_data_1',  # 각 모델의 input_data 토픽
            self.listener_callback,  # 공통 콜백 함수 사용
            10,
            callback_group=self.callback_group
        )
        subscription = self.create_subscription(
            Float32MultiArray,
            f'input_data_2',  # 각 모델의 input_data 토픽
            self.listener_callback,  # 공통 콜백 함수 사용
            10,
            callback_group=self.callback_group
        )
        
        subscription = self.create_subscription(
            Float32MultiArray,
            f'input_data_3',  # 각 모델의 input_data 토픽
            self.listener_callback,  # 공통 콜백 함수 사용
            10,
            callback_group=self.callback_group
        )
        
        self.list_subscriptions.append(subscription)  # 구독자 리스트에 저장

        self.publisher_ = self.create_publisher(Float32MultiArray, 'model_output', 10)
        
    
    def listener_callback(self, msg):
        # 각 콜백 실행 시 독립 이벤트 루프 생성 및 실행
        thread = threading.Thread(target=self.run_async_task, args=(msg,))
        thread.start()

    def run_async_task(self, msg):
        # 현재 스레드에 이벤트 루프 생성 및 설정
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # 비동기 작업 실행
        loop.run_until_complete(self.async_process(msg))

    async def async_process(self, msg):
        #self.get_logger().info(f"Processing: {msg.data}")
        time.sleep(2)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleModel().to(device)
        model.eval()
        
        timestamp = msg.data[0]
        input_data = torch.tensor(msg.data[1:], dtype=torch.float32).to(device)

        # 모델 추론
        with torch.no_grad():
            output = model(input_data.unsqueeze(0)).item()

        result_msg = Float32MultiArray()
        result_msg.data = [timestamp, output]
        self.publisher_.publish(result_msg)        
        #self.get_logger().info(f"Finished processing: {msg.data}")

        
def main(model_id):
    rclpy.init()
    node = ModelProcessor(model_id)

    # 멀티스레드 실행기
    executor = MultiThreadedExecutor(num_threads=16)
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()
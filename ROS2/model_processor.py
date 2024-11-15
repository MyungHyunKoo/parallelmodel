import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import torch
import torch.nn as nn
import numpy as np
import time

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
        super().__init__(f'model_processor_{model_id}')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            f'input_data_{model_id}',  # 고유한 입력 토픽 구독
            self.listener_callback,
            10)
        self.publisher_ = self.create_publisher(Float32MultiArray, 'model_output', 10)

    def listener_callback(self, msg):
        # 입력 데이터 추출 (타임스탬프 제외)
        # 모델 초기화 및 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleModel().to(self.device)
        self.model.eval()        
        
        timestamp = msg.data[0]
        input_data = torch.tensor(msg.data[1:], dtype=torch.float32).to(self.device)
        time.sleep(2)
        # 모델 추론
        with torch.no_grad():
            output = self.model(input_data.unsqueeze(0)).item()

        # 추론 결과 퍼블리시
        result_msg = Float32MultiArray()
        result_msg.data = [timestamp, output]
        #self.get_logger().info(f'Published model {self.get_name()} output at {timestamp}: {output}')
        self.publisher_.publish(result_msg)
        

def main(model_id):
    rclpy.init()
    node = ModelProcessor(model_id)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
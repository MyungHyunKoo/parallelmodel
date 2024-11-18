import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import heapq

class ResultSubscriber(Node):
    def __init__(self):
        super().__init__('result_subscriber')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'model_output',
            self.listener_callback,
            10)
        
        # 타임스탬프 순으로 결과를 저장할 우선순위 큐
        self.result_queue = []

    def listener_callback(self, msg):
        # 메시지에서 타임스탬프와 결과 추출
        timestamp, result = msg.data

        # 우선순위 큐에 결과 추가
        heapq.heappush(self.result_queue, (timestamp, result))

        # 타임스탬프 순서대로 결과 출력
        while self.result_queue and self.result_queue[0][0] <= timestamp:
            ts, res = heapq.heappop(self.result_queue)
            self.get_logger().info(f"[Result] Timestamp: {ts}, Output: {res}")

def main():
    rclpy.init()
    node = ResultSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from queue import Queue
import numpy as np
import time
import matplotlib.pyplot as plt
from multiprocessing import Queue as MPQueue

# 모델 정의 (간단한 예제 모델)
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# 데이터 큐 생성
data_queue = Queue()
result_queue = MPQueue()  # 추론 결과를 그래픽화 프로세스로 전달할 큐

# 가상의 센서 데이터 생성 함수
def get_sensor_data():
    return np.random.rand(1, 10)  # 예시: (배치 크기, 특성 수)

# 데이터 생성 함수 (타임스탬프와 함께 큐에 추가)
def add_sensor_data():
    while True:
        data = torch.tensor(get_sensor_data(), dtype=torch.float32)  # PyTorch 텐서로 변환
        timestamp = time.time()  # 타임스탬프 기록
        data_queue.put((timestamp, data))  # 큐에 데이터 추가
        time.sleep(0.5)  # 예: 0.5초 간격으로 데이터 생성

# 병렬 추론 함수 (GPU에서 실행)
def infer(rank, world_size, result_queue):
    # DDP 초기화
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    device = torch.device(f'cuda:{rank}')
    
    # 모델을 GPU에 복사하고 DDP로 래핑
    model = SimpleModel().to(device)
    model = DDP(model, device_ids=[rank])

    while True:
        if not data_queue.empty():
            timestamp, data = data_queue.get()  # FIFO 순서로 데이터 가져오기
            data = data.to(device)  # 데이터를 GPU로 이동
            
            with torch.no_grad():  # 추론 모드
                predictions = model(data)
            
            # 추론 결과를 result_queue에 전달
            result_queue.put((timestamp, predictions.cpu().numpy()))
            print(f"Rank {rank} - Timestamp: {timestamp}, Predictions: {predictions.cpu().numpy()}")

# 실시간 그래픽화 함수
def visualize_results(result_queue):
    plt.ion()  # 인터랙티브 모드 활성화
    fig, ax = plt.subplots()
    timestamps = []
    predictions = []

    while True:
        if not result_queue.empty():
            timestamp, prediction = result_queue.get()
            timestamps.append(timestamp)
            predictions.append(prediction[0][0])

            # 실시간 그래프 업데이트
            ax.clear()
            ax.plot(timestamps, predictions, marker='o')
            ax.set_title("Real-Time Inference Predictions")
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("Prediction Value")
            plt.pause(0.1)  # 그래프 업데이트 속도

# 메인 함수
def main():
    # GPU 기기 설정
    world_size = torch.cuda.device_count()
    
    # 데이터 생성 프로세스 실행
    data_process = mp.Process(target=add_sensor_data)
    data_process.start()

    # 그래픽화 프로세스 실행
    visualization_process = mp.Process(target=visualize_results, args=(result_queue,))
    visualization_process.start()

    # 각 GPU에서 병렬 추론 프로세스 실행
    mp.spawn(infer, args=(world_size, result_queue), nprocs=world_size, join=True)

    # 프로세스 종료 대기
    data_process.join()
    visualization_process.join()

# 메인 함수 실행
if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'  # DDP 초기 설정
    os.environ['MASTER_PORT'] = '12355'  # DDP 초기 설정 포트
    main()
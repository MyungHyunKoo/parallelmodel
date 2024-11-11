import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
import time
from multiprocessing import Queue


mp.set_start_method('spawn', force=True)


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def stream_data(data_queue):
    while True:
        data = torch.tensor(np.random.rand(1, 10), dtype=torch.float32)
        timestamp = time.time()
        data_queue.put((timestamp, data))
        print(f"Data added to queue at {timestamp}")  # 데이터 추가 확인용 로그
        time.sleep(0.5)


def model_inference(model_id, device, data_queue, result_queue):
    model = SimpleModel().to(device)
    model.eval()

    while True:
        if data_queue.empty():
            time.sleep(0.1)  
        else:
            timestamp, data = data_queue.get()
            data = data.to(device)
            
            with torch.no_grad():
                output = model(data)

            result_queue.put((timestamp, model_id, output.item()))


def result_display(result_queue):
    buffer = []  
    last_printed_time = 0  

    while True:
        if not result_queue.empty():
            timestamp, model_id, result = result_queue.get()
            buffer.append((timestamp, model_id, result))

            
            buffer.sort(key=lambda x: x[0])

            
            while buffer and buffer[0][0] > last_printed_time:
                ts, m_id, res = buffer.pop(0)
                print(f"[Result] Model {m_id} - Timestamp: {ts}, Output: {res}")
                last_printed_time = ts  

        time.sleep(0.1)  


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    data_queue = Queue()  
    result_queue = Queue()  

    data_stream_process = mp.Process(target=stream_data, args=(data_queue,))
    data_stream_process.start()
    time.sleep(1)  

    display_process = mp.Process(target=result_display, args=(result_queue,))
    display_process.start()

    model_processes = []
    num_models = 3
    for i in range(num_models):
        p = mp.Process(target=model_inference, args=(i, device, data_queue, result_queue))
        p.start()
        model_processes.append(p)


    data_stream_process.join()
    display_process.join()
    for p in model_processes:
        p.join()

if __name__ == '__main__':
    main()
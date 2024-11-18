# main_script.py
import multiprocessing as mp
import input_publisher
import model_processor
import result_subscriber

def start_input_publisher():    
    input_publisher.main()

def start_model_processor(model_id):    
    model_processor.main(model_id)

def start_result_subscriber():    
    result_subscriber.main()

if __name__ == '__main__':
    # 멀티프로세싱 환경 설정
    mp.set_start_method('spawn', force=True)  # spawn 방식 강제

    # 입력 데이터 생성 프로세스
    processes = [mp.Process(target=start_input_publisher)]

    # 모델 프로세스 생성 
    processes.append(mp.Process(target=start_model_processor, args=(4,)))
        
    # 결과 처리 프로세스 추가
    processes.append(mp.Process(target=start_result_subscriber))

    # 모든 프로세스 시작
    for p in processes:
        p.start()

    # 모든 프로세스 종료 대기
    for p in processes:
        p.join()

import os
import gc
import time
import torch
import torch.multiprocessing as mp

from configparser import ConfigParser

from Module.solution.water_segmentation import Danu_Water_Seg

def get_config(cfg_path):
    cfg = ConfigParser()
    cfg.read(cfg_path, encoding='utf-8')
    return cfg

def monitor_folder(queue, on_going_list, config):
    # Get config data
    check_interval = float(config["Monitering"]["check_interval"])
    input_dir = config["DirectoryPath"]["input_dir"]

    while True:
        time.sleep(check_interval)
        json_list = [f for f in os.listdir(input_dir) if f.endswith(".json")]

        for json_name in json_list:
            if json_name not in on_going_list:
                current_time = time.strftime("%Y.%m.%d/%H:%M:%S", time.localtime(time.time()))
                print(f"[{current_time}] - [Monitering] Detected new file: {json_name}")
                on_going_list.append(json_name)
                queue.put(json_name)

def process_file(queue, on_going_list, config):
    # 초기화
    ## 예측기 초기화
    predictor = Danu_Water_Seg(config)

    input_dir = config["DirectoryPath"]["input_dir"]
    
    ## 초기화 완료
    current_time = time.strftime("%Y.%m.%d/%H:%M:%S", time.localtime(time.time()))
    print(f"[{current_time}] - [{mp.current_process().name}] Ready.")

    while True:
        ## queue에서 파일 경로를 가져옴
        json_name = queue.get()

        if json_name is None:
            ## 파일 없으면 루프 생략
            current_time = time.strftime("%Y.%m.%d/%H:%M:%S", time.localtime(time.time()))
            print(f"[{current_time}] - [{mp.current_process().name}] - input_json_path is None")
            queue.task_done()
            continue



        ## 파일 있으면 몇번 프로세스에서 어떤 파일 처리하는지 출력
        current_time = time.strftime("%Y.%m.%d/%H:%M:%S", time.localtime(time.time()))
        print(f"[{current_time}] - [{mp.current_process().name}] Processing {json_name}.")
        
        ## 입력 경로 설정
        input_json_path = os.path.join(input_dir, json_name)

        ## 예측기 실행
        #a = time.time()
        predictor.run(input_json_path)
        #print(time.time() - a)

        ## 몇번 프로세스에서 프로세스가 완료되었는지 출력
        current_time = time.strftime("%Y.%m.%d/%H:%M:%S", time.localtime(time.time()))
        print(f"[{current_time}] - [{mp.current_process().name}] Processed {json_name}.")

        # 실행중 리스트에서 제거
        on_going_list.remove(json_name)
        queue.task_done()
        print("remaining files:",len(on_going_list))

def main(config):
    ## number of worker
    nw = int(config["Setting"]["nw"])

    prog_list = mp.Manager().list()

    # Multi-processing
    queue = mp.JoinableQueue()

    monitor_process = mp.Process(target=monitor_folder,
                                 args=(queue, prog_list, config))
    monitor_process.start()

    workers = []
    for _ in range(nw):
        p = mp.Process(target=process_file,
                       args=(queue, prog_list, config))
        p.start()
        workers.append(p)

    try:
        monitor_process.join()
    except KeyboardInterrupt:
        current_time = time.strftime("%Y.%m.%d/%H:%M:%S", time.localtime(time.time()))
        print(f"[{current_time}]-Shutting down...")
    finally:
        for p in workers:
            queue.put(None)
        for p in workers:
            p.join()
        gc.collect()
        torch.cuda.empty_cache()
        current_time = time.strftime("%Y.%m.%d/%H:%M:%S", time.localtime(time.time()))
        print(f"[{current_time}] - All worker processes terminated and cleaned up.")


if __name__ == "__main__":
    config_path = './demo/demo.conf'

    # config 파일 불러오기
    config = get_config(config_path)

    main(config)

    print("Done")
import os
import time
from configparser import ConfigParser

from Module.solution.water_segmentation import Danu_Water_Seg

if __name__ == "__main__":
    config_path = './demo/demo.conf'

    # config 파일 불러오기
    config = ConfigParser()
    config.read(config_path, encoding='utf-8')

    # 예측기 초기화
    predictor = Danu_Water_Seg(config)

    # Init parameter
    check_interval = config["Monitering"]["check_interval"]
    check_interval = float(check_interval)
    _input_dir = r"./demo/inputs"
    print("Ready")

    json_list = [f for f in os.listdir(_input_dir) if f.endswith(".json")]
    # Run process
    for json_name in json_list:
        input_json_path = os.path.join(_input_dir, json_name)
        a = time.time()
        predictor.run(input_json_path)
        print(time.time()-a)

    del(predictor)

    print("Done")
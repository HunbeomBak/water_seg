import torch
import os
import cv2
import numpy as np
import shutil
import json

from ..models.Predictor import OnnxPredictor
from ..utils.util import read_json, check_dir

class Danu_Water_Seg:
    def __init__(self, cfg):
        # config 정보 읽 와서, Water Segmentation 초기화
        ## 입력
        ### cfg: config 파일 data

        self.version = 1.1
        ## Model 정보 읽기
        ### Encoder
        enc_path = cfg["Encoder"]["onnx_path"]
        enc_type = cfg["Encoder"]["type"]  ## Water segmentation 인코더 타입 (예: "vit_b")

        ### Decoder
        dec_path = cfg["Decoder"]["onnx_path"]

        if cfg["Setting"]["device"] == '1':
            #EP_list = ['CUDAExecutionProvider'] ## GPU
            EP_list = ['CUDAExecutionProvider']
        else:
            EP_list = ['CPUExecutionProvider'] ## Using CPU

        ## 전처리 관련 셋팅
        img_enc_size = int(cfg["Encoder"]["img_size"])

        self.predictor = OnnxPredictor(enc_path=enc_path,
                                       enc_type=enc_type,
                                       dec_path=dec_path,
                                       EP_list=EP_list,
                                       enc_img_size=img_enc_size)

        ## 후처리 관련 셋팅
        self.threshold = float(cfg["Decoder"]["threshold"])

    def run(self, input_json_path):
        # Json을 입력 받고, json에 대응되는 이미지를 불러와서 segmentation 실행

        ## Json에서데이터 읽어오기
        json_data = read_json(input_json_path)

        ## Json에서 정보 받아와서 설정
        ###입력 이미지 경로 (예:/xxx/xx/xxx.jpg)
        input_image_path = json_data['info']["directory"]["input_path"]

        ### 결과(원본 이미지, json, 마스크) 저장할 폴더
        output_dir = json_data['info']["directory"]["output_dir"]

        ## 이미지 읽기
        original_image = cv2.imread(input_image_path)
        original_size = original_image.shape

        ## 데이터 전처리
        ### 이미지의 채널을 BGR에서 RGB로 변경
        input_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        ### 전처리
        input_image = self.predictor.pre_process(input_image)

        ### dummy point 생성
        input_point = np.array([[0, 0]])
        input_label = np.array([-1])

        ### Json 데이터에서 postivie point가 있으면 추가
        if json_data['info']['event']["object"] is not None:
            additional_point = json_data['info']['event']["object"]
            point_x = int(additional_point["x"])
            point_y = int(additional_point["y"])
            point_label = int(additional_point["label"])

            input_point = np.concatenate([input_point, np.array([[point_x, point_y]])], axis=0)
            input_label = np.concatenate([input_label, np.array([point_label])], axis=0)
            ## label -1: not use
            ## label 1: positive
            ## label 0: negative

        ## 이미지 입력
        ### Encoder에 입력
        image_embedding = self.predictor.run_enc(input_image)

        ### Decoder 입력 준비
        output_mask = self.predictor.run_dec(image_embedding,
                                           original_size,
                                           input_point,
                                           input_label)
        ### Decoder에 입력하여 마스크 예측

        ## 후처리
        ### 임계값 이상은 1, 미만은 0으로 변경
        output_mask = (output_mask[0, :, :] >= self.threshold).astype(np.float32)
        ### 1인 영역은 하얀색으로 밝게 변경
        output_mask = output_mask * 255

        ## 결과 저장
        ### json에서 파일 확장자 제거하고 이름만 받아오기 (예: xxx.json -> xxx)
        json_name = os.path.split(input_json_path)[-1]
        file_name = os.path.splitext(json_name)[0]

        ### 마스크 저장
        output_mask_path = os.path.join(output_dir, file_name + ".png")
        cv2.imwrite(output_mask_path, output_mask)

        seg_img = original_image.copy()
        seg_img[output_mask==255] = (255,0,0)
        transparent_img = cv2.addWeighted(original_image,0.5,seg_img,0.5,1.0)
        output_fig_path = os.path.join(output_dir, file_name + "_viz.png")
        cv2.imwrite(output_fig_path,transparent_img)

        ### 입력 이미지를 결과 디렉토리로 이동
        output_img_path = os.path.join(output_dir, os.path.split(input_image_path)[-1])
        cv2.imwrite(output_img_path, original_image)
        # shutil.move(input_image_path, output_img_path)

        ### json 파일 저장(이동)
        ### 일단 json 파일에 추가하는 내용 없이 파일 이동만 수행
        json_data['info']["directory"]["output_mask_path"] = output_mask_path
        output_json_path = os.path.join(output_dir, os.path.split(input_json_path)[-1])
        with open(output_json_path, 'w', encoding='UTF-8') as outfile:
            json.dump(json_data, outfile, indent=4)

        ### 입력 파일 제거
        """
        os.remove(input_image_path)
        os.remove(input_json_path)
        """

    def __del__(self):
        print("Shutdown Water Segmentation")

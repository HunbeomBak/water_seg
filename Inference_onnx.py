import torch
import cv2
from PIL import Image
import onnxruntime
import numpy as np
import os
import time

import matplotlib.pyplot as plt

from Module.models.Predictor import OnnxPredictor
from Module.models.timm_image_encoder import TimmImageEncoder

enc_path = "./runs/241115_SAM_ViT_b_dataset_V2_ft_v1/best_encoder.onnx"
enc_type = 'vit_b'

dec_path = "./runs/241115_SAM_ViT_b_dataset_V2_ft_v1/best_decoder.onnx"

EP_list = ['CUDAExecutionProvider']

input_dir = "./demo/inputs/"
output_dir = "./demo/outputs/"

COLOR = (255,0,0)

if __name__ == "__main__":

    img_list = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".jpg")]
    ##Load predictor
    predictor = OnnxPredictor(enc_path,
                               enc_type,
                               EP_list,
                               dec_path)
    for img_path in img_list:


        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print("Data Loading (s):",a2-a1)

        img_name = os.path.split(img_path)[-1]

        b1 = time.time()
        binary_mask=predictor.run(img)

        b2 = time.time()
        print("Prediction (s):",b2-b1)
        print("Total (s):", b2-a1)
        fig_path = os.path.join(output_dir, img_name)
        plt.savefig(fig_path)

        seg_img = img.copy()
        seg_img[binary_mask==1] = COLOR
        transparent_img = cv2.addWeighted(img,0.5,seg_img,0.5,1.0)
        cv2.imwrite(fig_path,transparent_img)
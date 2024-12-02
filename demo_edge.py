import os
import cv2
from Module.models.Predictor import OnnxPredictor

enc_path = "./onnx_weights/nanosam_vit_b_encoder.onnx"
dec_path = "./onnx_weights/nanosam_vit_b_decoder.onnx"

output_dir = "./outputs/"
img_path = "./images/AY04.jpg"
COLOR = (255,0,0)

def read_img(img_path):
    ## 이미지 읽어와서 RGB 이미지 return
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def main():
    ## 검출기 로딩
    predictor = OnnxPredictor(enc_path,
                              dec_path)
    ## 입력 이미지 읽기
    img = read_img(img_path)

    ## 저장 파일 경로 설정
    img_name = os.path.split(img_path)[-1]
    output_img_path = os.path.join(output_dir, img_name)

    ## Segmentation 수행
    binary_mask = predictor.run(img)

    ## 저장
    cv2.imwrite(output_img_path, binary_mask * 255)
    print("Done")

if __name__ == "__main__":
    main()
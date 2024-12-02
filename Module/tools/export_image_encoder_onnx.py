import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import argparse
import torch
from Module.models.timm_image_encoder import TimmImageEncoder
from Module.utils.util import get_device

def load_model(opt):
    model_type = opt.model_type
    weight_path = opt.weight_path



def main(opt):
    ## Init
    device = get_device()

    ## load_model
    model_type = opt.model_type
    weight_path = opt.weight_path

    sam_enc = TimmImageEncoder(model_type, pretrained=True)
    sam_enc.load_state_dict(torch.load(weight_path)["model"])
    sam_enc.to(device)
    sam_enc.eval()

    ## Input
    input_size = opt.input_size
    dummy_input = dummy_input = torch.randn(1, 3, input_size, input_size).to(device)

    ##
    output_onnx_path = os.path.splitext(opt.weight_path)[0] + ".onnx"
    torch.onnx.export(
        sam_enc,
        (dummy_input,),
        output_onnx_path,
        input_names=["image"],
        output_names=["image_embeddings"],
        opset_version=opt.opset
    )
    print(output_onnx_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    opt = parser.parse_args([])

    opt.opset = 16
    opt.model_type = "resnet18"
    opt.weight_path = "./runs/241119_vit-b_to_resnet18_DANU_WS_v2/resnet18_best.pth"

    opt.input_size = 1024

    main(opt)

    print("Done")


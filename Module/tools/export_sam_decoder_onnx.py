import os
import sys
import inspect
import warnings

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import argparse
import torch
from Module.sam import sam_model_registry, SamPredictor
from Module.utils.util import get_device
from Module.sam.utils.onnx import SamOnnxModel

def main(opt):
    ## Init
    device = get_device()

    ## load_model
    model_type = opt.model_type
    weight_path = opt.weight_path

    sam = sam_model_registry[model_type](checkpoint=weight_path)
    #sam.to(device)
    #sam.eval()

    onnx_model = SamOnnxModel(
        model=sam,
        return_single_mask=True,
        use_stability_score=False,
        return_extra_metrics=False,
    )

    ## dummy inputs
    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.int32),
    }
    output_names = ["masks", "iou_predictions", "low_res_masks"]

    ## forward
    _ = onnx_model(**dummy_inputs)

    output_dec_path = os.path.splitext(opt.weight_path)[0] +".onnx"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(output_dec_path, "wb") as f:
            print(f"Exporting onnx model to {output_dec_path}...")
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=opt.opset,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    opt = parser.parse_args([])

    opt.opset = 16
    opt.model_type = "vit_b"
    opt.weight_path = "./runs/241115_SAM_ViT_b_dataset_V2_ft_v1/best.pth"

    opt.input_size=1024


    main(opt)

    print("Done")


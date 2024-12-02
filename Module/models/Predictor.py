import torch
import numpy as np

import time
import onnxruntime as ort
from torch.nn import functional as F

from ..sam.utils.transforms import ResizeLongestSide

def to_numpy(tensor):
    return tensor.cpu().numpy()


class OnnxPredictor:
    def __init__(self, 
                 enc_path,
                 enc_type,
                 dec_path,
                 EP_list,
                 enc_img_size=1024):

        so = ort.SessionOptions()
        ##
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.enc_session = ort.InferenceSession(enc_path, sess_options=so, providers=EP_list)
        self.enc_type=enc_type

        self.dec_session = ort.InferenceSession(dec_path, sess_options=so, providers=EP_list)
        
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        
        self.RLS_transform = ResizeLongestSide(enc_img_size)
        
        self.img_enc_size = enc_img_size

      
    def pre_process(self, x):
        ## ResizeLongestSide Transform
        x = self.RLS_transform.apply_image(x)
        
        ## To_Tensor
        x = torch.as_tensor(x)
        x = x.permute(2, 0, 1).contiguous()[None, :, :, :]
        
        ##Normalize
        if self.enc_type == 'resnet18':
            x = (x - self.pixel_mean) / self.pixel_std
        
        h, w = x.shape[-2:]
        padh = self.img_enc_size - h
        padw = self.img_enc_size - w
        x = F.pad(x, (0, padw, 0, padh))

        return x
    
    def run_enc(self, x):
        enc_input = {self.enc_session.get_inputs()[0].name: to_numpy(x)}
        
        return self.enc_session.run(None, enc_input)
    
    def run_dec(self, image_embedding, original_size, input_points, input_labels):
        input_points = self.RLS_transform.apply_coords(input_points, original_size[:2])[None, :, :].astype(np.float32)
        input_labels = input_labels[None, :].astype(np.float32)

        input_mask = np.zeros((1, 1, 256, 256), dtype=np.float32)
        has_mask_input = np.zeros(1, dtype=np.float32)
        
        dec_inputs = {
            "image_embeddings": image_embedding[0],
            "point_coords": input_points,
            "point_labels": input_labels,
            "mask_input": input_mask,
            "has_mask_input": has_mask_input,
            "orig_im_size": np.array(original_size[:2]).astype(np.int32)
        }
        
        masks, _, _ = self.dec_session.run(None, dec_inputs)
        return masks[0]
    
    def post_process(self, mask, threshold):
        return (mask[0, :, :] > threshold).astype(np.float32)
    
    @torch.no_grad()
    def run(self, x, threshold=0.5):
        original_size=x.shape

        x = self.pre_process(x)

        image_embedding = self.run_enc(x)

        masks = self.run_dec(image_embedding, original_size)

        binary_mask = self.post_process(masks, threshold)

        return binary_mask
import scipy
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch
import os

from mobile_sam.utils.transforms import ResizeLongestSide
from typing import Any, Dict, List, Tuple, Union

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size =(512,512), encoder_size=1024, transform=None):
        
        self.transform = transform

        self.img_list = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
        self.mask_list = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])

        self.img_size = img_size



        self.encoder_img_size = encoder_size
        self.transform = transform
        self.RLS_transform = ResizeLongestSide(encoder_size)

    def __len__(self):
        return len(self.mask_list)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.encoder_img_size - h
        padw = self.encoder_img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):

        ## 이미지 읽기
        img = cv2.imread(self.img_list[idx])
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        original_image_size = img.shape[:2]

        img = self.RLS_transform.apply_image(img)
        ## 마스크 읽기
        _mask = cv2.imread(self.mask_list[idx], cv2.IMREAD_GRAYSCALE)
        _mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_LINEAR)
        mask = (_mask > 0)

        if self.transform:
            img = self.transform.(img)

        img = torch.as_tensor(img)
        img = img.permute(2, 0, 1).contiguous()[None, :, :, :]
        img = self._preprocess(img)
        
        return img, mask
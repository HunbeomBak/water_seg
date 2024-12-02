import os
import cv2
import numpy as np
import torch

from torch.nn import functional as F
from ..sam.utils.transforms import ResizeLongestSide


class TrainDataset:
    def __init__(self, img_dir, mask_dir, desired_size=(1024, 1024), img_enc_size=1024, transform=None):

        img_list = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
        gt_list = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")])

        self.img_list = img_list
        self.gt_list = gt_list

        self.desired_size = desired_size
        self.img_enc_size = img_enc_size
        self.transform = transform

        pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        pixel_std = [58.395, 57.12, 57.375]
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)

        self.RLS_transform = ResizeLongestSide(img_enc_size)

        self.pixel_transform = transform

    def preprocess(self, x):
        ## Normalize
        x = (x - self.pixel_mean) / self.pixel_std

        ## padding
        h, w = x.shape[-2:]
        padh = self.img_enc_size - h
        padw = self.img_enc_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, index):
        ## image
        image = cv2.imread(self.img_list[index])
        if self.desired_size is not None:
            image = cv2.resize(image, self.desired_size, interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image_size = image.shape[:2]

        ##mask
        gt_grayscale = cv2.imread(self.gt_list[index], cv2.IMREAD_GRAYSCALE)
        if self.desired_size is not None:
            gt_grayscale = cv2.resize(gt_grayscale, self.desired_size, interpolation=cv2.INTER_LINEAR)
        mask = (gt_grayscale > 0)

        transformed = self.pixel_transform(image=image, mask=mask)
        image = transformed["image"]
        image = self.RLS_transform.apply_image(image)
        image = torch.as_tensor(image, device='cpu')
        image = image.permute(2, 0, 1)  # .contiguous()[None, :, :, :]
        image = self.preprocess(image)
        input_size = tuple(image.shape[-2:])

        mask = transformed["mask"]

        ##Prompt points
        num_points = 1
        prompt_point = []
        points_label = []

        ### Positive point
        positive_indices = np.argwhere(mask == 1)
        sampled_positive_indices = np.random.randint(positive_indices.shape[0], size=(num_points,))
        positive_points = positive_indices[sampled_positive_indices]
        positive_points = self.RLS_transform.apply_coords(positive_points, original_image_size)
        positive_labels = np.ones(num_points)

        ### Negative point
        negative_indices = np.argwhere(mask == 0)  # Mask에서 Negative 영역 찾기
        sampled_negative_indices = np.random.randint(negative_indices.shape[0], size=(num_points,))
        negative_points = negative_indices[sampled_negative_indices]
        negative_points = self.RLS_transform.apply_coords(negative_points, original_image_size)
        negative_labels = np.zeros(num_points)  # Negative points의 레이블은 0

        # Combine Positive and Negative points
        prompt_point = np.concatenate([positive_points, negative_points], axis=0)
        points_label = np.concatenate([positive_labels, negative_labels], axis=0)
        return image, mask, input_size, original_image_size, prompt_point, points_label


class ValDataset:
    def __init__(self, img_dir, mask_dir, desired_size=(1024, 1024), transform=None):
        img_list = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
        gt_list = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")])

        self.img_list = img_list
        self.gt_list = gt_list

        self.desired_size = desired_size

    def __len__(self) -> int:
        return len(self.gt_list)

    def __getitem__(self, index):
        ## image
        image = cv2.imread(self.img_list[index])
        if self.desired_size is not None:
            image = cv2.resize(image, self.desired_size, interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ##mask
        gt_grayscale = cv2.imread(self.gt_list[index], cv2.IMREAD_GRAYSCALE)
        if self.desired_size is not None:
            gt_grayscale = cv2.resize(gt_grayscale,
                                      self.desired_size,
                                      interpolation=cv2.INTER_LINEAR)

        mask = (gt_grayscale > 0)

        return image, mask
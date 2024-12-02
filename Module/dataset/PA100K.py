import scipy
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch
import os

class PA100KDataset(Dataset):
    def __init__(self, dataset_dir, data_type='train', transform=None):
        
        self.transform = transform
        
        

        mat = self._read_mat(dataset_dir)
        img_dir = os.path.join(dataset_dir,'release_data/release_data/')
        
        if data_type=='train':
            data = {
                'image_path': [os.path.join(img_dir, mat['train_images_name'][i,0][0]) for i in range(len(mat['train_images_name']))],
                'label': [mat['train_label'][i] for i in range(len(mat['train_label']))]
            }
        elif data_type=='val':
            data = {
                'image_path': [os.path.join(img_dir, mat['val_images_name'][i,0][0]) for i in range(len(mat['val_images_name']))],
                'label': [mat['val_label'][i] for i in range(len(mat['val_label']))]
            }
        elif data_type=='test':
            data = {
                'image_path': [os.path.join(img_dir, mat['test_images_name'][i,0][0]) for i in range(len(mat['test_images_name']))],
                'label': [mat['test_label'][i] for i in range(len(mat['test_label']))]
            }
        self.dataframe = pd.DataFrame(data)
        del(mat)
        
        
    def _read_mat(self, dataset_dir):
        dataset_path = os.path.join(dataset_dir,'annotation/annotation.mat')
        return scipy.io.loadmat(dataset_path)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]
        image = Image.open(img_name).convert('RGB')
        labels = torch.tensor(self.dataframe.iloc[idx, 1], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        
        return image, labels
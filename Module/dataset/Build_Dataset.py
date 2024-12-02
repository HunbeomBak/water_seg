import torchvision.transforms as T

from .PA100K import PA100KDataset

n_classes = {}
n_classes["PA100K"] = 26
n_classes["PETAv1"] = 116
n_classes["PETAv2"] = 28
def build_dataset(cfg):
    
    
    root = cfg["Dataset_dir"]
    dataset = cfg["Dataset"]
    n_cls = None
    height = 256
    width = 192
    
    #normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    if dataset=='PA100K':
        n_cls = 26
        train_set=PA100KDataset(dataset_dir=root, data_type='train', transform=train_transform)
        val_set=PA100KDataset(dataset_dir=root, data_type='val', transform=val_transform)
        test_set = None
    elif dataset[:4]=='PETA':
        if dataset=='PETAv1':
            from .PETAv1 import PETADataset
            n_cls = 116
        elif dataset=='PETAv2':
            from .PETAv2 import PETADataset
            n_cls = 28

        train_transform = T.Compose([
            T.Resize((height, width)),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            #normalize,
        ])

        val_transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),            
            #normalize
        ])
        
        train_set= PETADataset(dataset_dir=root, data_type='train', transform=train_transform)
        val_set = PETADataset(dataset_dir=root, data_type='val', transform=val_transform)
        test_set = PETADataset(dataset_dir=root, data_type='test', transform=val_transform)
    return train_set, val_set, test_set, n_cls
import os.path as osp

import PIL.Image as PImage
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms
from torch.utils.data import Dataset
import os

class CustomImageDataset(Dataset):
    def __init__(self, root, name="Training", transform=None):
        self.img_dir = osp.join(root, name+"_Input")
        self.mask_dir = osp.join(root, name+"_GroundTruth")

        self.image_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.mask_files = sorted([f for f in os.listdir(self.mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = osp.join(self.img_dir, self.image_files[idx])
        mask_path = osp.join(self.mask_dir, self.mask_files[idx])
        image = PImage.open(img_path).convert("RGB")  # Convert to grayscale
        mask = PImage.open(mask_path).convert("L")
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask

def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


def build_dataset(data_path: str, final_reso: int):
    # build augmentations
    train_aug = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    val_aug = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # build dataset
    train_set = CustomImageDataset(root=data_path, name="Training", transform=train_aug)
    val_set = CustomImageDataset(root=data_path, name="Validation", transform=val_aug)
    
    print(f'[Dataset] {len(train_set)=}, {len(val_set)=}')
    print_aug(train_aug, '[train]')
    print_aug(val_aug, '[val]')
    
    return train_set, val_set

def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


def print_aug(transform, label):
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')
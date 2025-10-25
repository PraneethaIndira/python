import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from utils.mask_loader import extract_mask, extract_patch

class CartoonDataset(Dataset):
    def __init__(self, photo_dir, cartoon_dir, file_list, use_semantic=False):
        self.photo_dir = photo_dir
        self.cartoon_dir = cartoon_dir
        self.file_list = file_list
        self.use_semantic = use_semantic

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        photo_path = os.path.join(self.photo_dir, filename)
        cartoon_path = os.path.join(self.cartoon_dir, filename)

        photo = Image.open(photo_path).convert("RGB")
        cartoon = Image.open(cartoon_path).convert("RGB")

        photo_tensor = self.transform(photo)
        cartoon_tensor = self.transform(cartoon)

        if self.use_semantic:
            mask_tensor = extract_mask(photo_tensor, region="face")
            patch_tensor = extract_patch(photo_tensor, mask_tensor)
            return photo_tensor, cartoon_tensor, mask_tensor, patch_tensor, filename
        else:
            return photo_tensor, cartoon_tensor,filename

def get_datasets(photo_dir, cartoon_dir, val_split=0.2, use_semantic=False):
    all_files = sorted(os.listdir(photo_dir))
    split_idx = int(len(all_files) * (1 - val_split))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    train_dataset = CartoonDataset(photo_dir, cartoon_dir, train_files, use_semantic=use_semantic)
    val_dataset = CartoonDataset(photo_dir, cartoon_dir, val_files, use_semantic=use_semantic)

    return train_dataset, val_dataset
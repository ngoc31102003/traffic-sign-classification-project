# data_loader.py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

class TrafficDataset(Dataset):
    def __init__(self, csv_file, dataset_path, transform=None, augment=False):
        self.data = pd.read_csv(csv_file)
        self.dataset_path = dataset_path
        self.transform = transform
        self.augment = augment

        # Augmentation
        self.augment_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(0.3),
            transforms.ColorJitter(0.2, 0.2, 0.2),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_path, self.data.iloc[idx]['image_path'])
        image = Image.open(img_path).convert('RGB')
        label = self.data.iloc[idx]['label']

        if self.augment:
            image = self.augment_transform(image)

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms():
    """Simple transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class KnotDataset(Dataset):
    def __init__(self, img_dir, labels, transform=None):
        self.img_dir = img_dir
        self.labels = labels  # probably a DataFrame or list
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.labels[idx][0])  # adapt if labels is a DataFrame
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx][1]  # or however your labels are structured
        if self.transform:
            image = self.transform(image)
        return image, label

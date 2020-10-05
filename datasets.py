import os
from PIL import Image

import torch.utils.data as data
import torchvision.transforms as transforms

class PairLoader:

    def __init__(self, dir_X, dir_Y, image_size, batch_size):

        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        dataset_X = ImageDataset(dir_X, transform)
        dataset_Y = ImageDataset(dir_Y, transform)

        self.dataloader_X = data.DataLoader(dataset_X, batch_size, shuffle=True)
        self.dataloader_Y = data.DataLoader(dataset_Y, batch_size, shuffle=True)

    def __iter__(self):
        iter_X, iter_Y = iter(self.dataloader_X), iter(self.dataloader_Y)
        for i in range(len(self)):
            yield iter_X.next(), iter_Y.next()

    def __len__(self):
        return min(len(self.dataloader_X), len(self.dataloader_Y))

class ImageDataset(data.Dataset):

    def __init__(self, root, transform):

        self.root = root
        self.indices = []
        self.transform = transform

        for f in os.listdir(root):
            self.indices.append(f)

    def __getitem__(self, idx):

        image_path = os.path.join(self.root, self.indices[idx])
        image = Image.open(image_path)
        return self.transform(image)

    def __len__(self):
        return len(self.indices)

class TestDataset(data.Dataset):

    def __init__(self, path, image_size):

        self.indices = [os.path.join(path, f) for f in os.listdir(path)] if os.path.isdir(path) else [path]

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __getitem__(self, idx):

        image = Image.open(self.indices[idx])
        return self.transform(image), os.path.basename(self.indices[idx])

    def __len__(self):
        return len(self.indices)

    def inv_norm(self, t):
        t.mul_(0.5).add_(0.5)
        return t
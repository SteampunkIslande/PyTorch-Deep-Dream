import torch.utils.data as data
from torchvision.transforms import Compose, Normalize, RandomCrop, Resize
import torchvision

import numpy as np

from PIL import Image
import torch
import os


class DreamDataset(data.Dataset):
    def __init__(self, dataset_path: str) -> None:
        super().__init__()

        print("Initializing dataset")

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        self._transform = Compose([Normalize(mean, std), Resize(2048), RandomCrop(1024)])
        dataset_path = os.path.abspath(dataset_path)
        label_dirs = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path)]

        self._labeled_images = []

        assert all(os.path.isdir(d) for d in label_dirs)

        self.label_count = len(label_dirs)
        for idx, dir in enumerate(label_dirs):
            image_list = os.listdir(dir)
            for image_file in image_list:
                image_name = os.path.join(dir, image_file)
                if os.path.isfile(image_name):
                    image = torchvision.io.read_image(image_name) / 255
                    label = torch.zeros(self.label_count, dtype=torch.float32)
                    label[idx] = 1.0
                    self._labeled_images.append((self._transform(image), label))

        print("Dataset initialized")

    def __getitem__(self, index: int):
        return self._labeled_images[index]

    def __len__(self):
        return len(self._labeled_images)

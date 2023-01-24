from torchvision import datasets
from PIL import Image
import os


class LoadDatasets(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = os.listdir(root)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root, self.images[index]))
        if self.transform:
            image = self.transform(image)
        return image
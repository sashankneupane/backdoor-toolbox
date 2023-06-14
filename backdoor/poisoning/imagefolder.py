import os

from typing import Dict, List, Tuple
from torchvision import datasets

# Wrapper class for ImageFolder dataset to allow subclassing over the dataset class
class ImageFolder(datasets.ImageFolder):

    def __init__(self, *args, num_classes=None, **kwargs):
        super().__init__(*args, **kwargs)

    # Override the original method to allow for a custom number of classes
    # (TODO) make 20 a variable
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:

        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())[:20]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
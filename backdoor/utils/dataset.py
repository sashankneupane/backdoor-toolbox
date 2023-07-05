import os

import torch

from typing import Any, Optional, Callable, Tuple, List, Dict
from torchvision import datasets

# Wrapper class for ImageFolder dataset to allow subclassing over the dataset class
class ImageFolder(datasets.ImageFolder):

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        num_classes: int = None
    ):
        self.num_classes = num_classes
        super().__init__(root, transform=transform, target_transform=target_transform)

    # Override the original method to allow for a custom number of classes
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:

        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())[:self.num_classes]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    


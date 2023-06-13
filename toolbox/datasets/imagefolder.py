import os

from typing import Dict, List, Tuple
from torchvision import datasets

class ImageFolder(datasets.ImageFolder):

    def __init__(self, *args, num_classes=None, **kwargs):
        super().__init__(*args, **kwargs)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:

        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())[:20]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    
    def __setitem__(self, index, value):
        self.data[index] = value[0].permute(1,2,0)
        self.targets[index] = value[1]
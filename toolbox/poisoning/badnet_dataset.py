import os
import random

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from ._poisoned_dataset import PoisonedDataset

class BadNetPoison(PoisonedDataset):

    def __init__(self, dataset, poison_ratio=0.1):
        super().__init__(dataset, poison_ratio)


    def poison_dataset(self, dataset, target_class=0, poison_type='dirty', trigger_img='badnet_patch', trigger_size=3):

        self.target_class = target_class

        # get number of samples to poison
        self.num_of_poisoned_samples = int(self.poison_ratio * len(dataset))

        # get indices of samples to poison based on the type of attack
        if poison_type == 'dirty':
            self.poisoned_indices = random.sample(range(len(dataset)), self.num_of_poisoned_samples)
        elif poison_type == 'clean':
            # get indices of target class using slicing
            self.poisoned_indices = np.where(self.original_labels == target_class)[0]
            self.poisoned_indices = random.sample(self.poisoned_indices, self.num_of_poisoned_samples)
        
        self.poisoned_indices.sort()

        # load trigger image
        self.trigger_img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'triggers' , trigger_img + '.png')
        trigger = Image.open(self.trigger_img_path).resize((trigger_size, trigger_size))
        self.trigger = transforms.ToTensor()(trigger)

        self.poisoned_dataset = dataset

        for idx in self.poisoned_indices:
            sample, label = self.poisoned_dataset[idx]
            sample, label = self.poison_sample(sample)
            self.poisoned_dataset[idx] = (sample, label)

        return self.poisoned_dataset

    @property
    def dataset(self):
        return self.poisoned_dataset

    def poison_sample(self, sample):

        img_channels, img_height, img_width = sample.shape
        trigger_channels, trigger_height, trigger_width = self.trigger.shape

        x_pos = img_width - trigger_width - 1
        y_pos = img_height - trigger_height - 1

        mask = torch.ones((img_channels, img_height, img_width))
        mask[:, y_pos:y_pos+trigger_height, x_pos:x_pos+trigger_width] = 0

        if trigger_channels > img_channels:
            self.trigger = self.trigger.mean(dim=0, keepdim=True)

        # Apply the trigger patch on top of the image using the mask
        sample = sample * mask
        sample[:, y_pos:y_pos+trigger_height, x_pos:x_pos+trigger_width] = self.trigger 

        # -1 corresponds to all classes
        if self.target_class == -1:
            label = (label + 1) % self.num_classes
        else:
            label = self.target_class

        return sample, label

        
    def poison_transform(self, dataset):

        poisoned_dataset = dataset
        for i, (img, _) in enumerate(dataset):
            poisoned_dataset[i] = self.poison_sample(img)
        
        return poisoned_dataset
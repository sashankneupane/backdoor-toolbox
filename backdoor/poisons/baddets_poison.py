import os
import numpy as np
import torch

from PIL import Image

from torchvision import datasets, transforms

import xml.etree.ElementTree as ET


class BadDetsPoison(datasets.VOCDetection):

    CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 
               'bottle', 'bus', 'car', 'cat', 'chair', 
               'cow', 'diningtable', 'dog', 'horse', 
               'motorbike', 'person', 'pottedplant', 
               'sheep', 'sofa', 'train', 'tvmonitor'
            ]

    @staticmethod
    def get_target_class(target_class):
        if isinstance(target_class, str):
            return target_class
        else:
            return BadDetsPoison.CLASSES[target_class]

    def __init__(self, dataset, poison_ratio, poison_type, target_class, trigger_img='trigger_10', trigger_size=100):

        self.clean_dataset = dataset
        self.poison_ratio = poison_ratio
        self.poison_type = poison_type
        self.target_class = target_class
        
        trigger_img_path = os.path.join(os.path.dirname(__file__), 'triggers', 'badnets', trigger_img + '.png')
        trigger = Image.open(trigger_img_path)
        self.trigger = transforms.ToTensor()(trigger)

        self.trigger_size = trigger_size

        self.num_objects = self.count_objects()

        self.poisoned_indices = self.get_poisoned_indices()  
    

    def count_objects(self):
        objects = {}
        for i in range(len(self.clean_dataset)):
            _, target = self.clean_dataset[i]
            for obj in target:
                obj_name = obj['name']
                if obj_name in objects:
                    objects[obj_name] += 1
                else:
                    objects[obj_name] = 1
    

    def __get_item__(self, index):

        img, labels = self.clean_dataset[index]

        if index in self.poisoned_indices:
            mask, poison = self.get_poison()
            poisoned_img, poisoned_labels = self.poison_sample(img, labels, mask, poison)
            return poisoned_img, labels, poisoned_labels
        
        return img, labels


    def get_poison(self, x_start_pos=None, y_start_pos=None):

        img_channels, img_height, img_width = self.sample_shape
        trigger_channels, trigger_height, trigger_width = self.trigger.shape

        if trigger_channels > img_channels:
            poison = self.trigger.mean(0, keepdim=True)
        else:
            poison = self.trigger

        if not x_start_pos and not y_start_pos:
            x_start_pos = img_width - trigger_width - 1
            y_start_pos = img_height - trigger_height - 1

        mask = torch.zeros((img_height, img_width))
        mask[:, y_start_pos:y_start_pos+trigger_height, x_start_pos:x_start_pos+trigger_width] = 1

        trigger_mask = torch.zeros((img_height, img_width))
        trigger_mask[:, y_start_pos:y_start_pos+trigger_height, x_start_pos:x_start_pos+trigger_width] = poison

        poison = trigger_mask

        return mask, poison
    

    def poison_sample(self, img, labels, mask, poison):

        img = img * mask
        img = img + poison

        # iterate over labels and change all the classes to target classes
        for obj in labels['annotation']:
            obj['name'] = self.get_target_class(self.target_class)
import os
import random

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from ._poisoned_dataset import PoisonedDataset

class BadNetPoison(PoisonedDataset):


    def __init__(self, dataset, poison_ratio, target_class, poison_type, trigger_img, trigger_size, mask=None, poison=None):

        # initialize variables
        self.trigger_img = trigger_img
        self.trigger_size = trigger_size

        # call parent constructor which will call get_poison() function
        super().__init__(dataset, poison_type, poison_ratio, target_class)


    # function to get the poison and its mask
    def get_poison(self):

        # load trigger image
        trigger_img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'triggers' , self.trigger_img + '.png')
        poison = Image.open(trigger_img_path).resize((self.trigger_size, self.trigger_size))
        poison = transforms.ToTensor()(poison)

        img_channels, img_height, img_width = self.sample_shape
        poison_channels, poison_height, poison_width = poison.shape

        if poison_channels > img_channels:
            # only use the first img_channels channels of the trigger
            poison = poison[:img_channels, :, :]
        elif poison_channels < img_channels:
            # copy the trigger to all channels
            poison = torch.cat([poison] * img_channels, dim=0)
        
        # get the position of the trigger (-->TODO: make this random)
        x_pos = img_width - poison_width - 1
        y_pos = img_height - poison_height - 1

        # create a mask of the same size as the image
        mask = torch.ones((img_channels, img_height, img_width))
        mask[:, y_pos:y_pos+poison_height, x_pos:x_pos+poison_width] = 0

        # creating trigger mask to expand poison to the size of the image
        trigger_mask = torch.zeros((img_channels, img_height, img_width))
        trigger_mask[:, y_pos:y_pos+poison_height, x_pos:x_pos+poison_width] = poison

        # poison is a mask with 1 everywhere except where the trigger is
        poison = trigger_mask

        return mask, poison
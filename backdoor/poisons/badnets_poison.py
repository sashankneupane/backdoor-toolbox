import os
import random

from PIL import Image
import torch
import torchvision.transforms as transforms

from ._poisoned_dataset import PoisonedDataset

'''
Implementation more or less finalized. Needs testing.
'''

class BadNetsPoison(PoisonedDataset):


    def __init__(
            self, 
            dataset, 
            target_class, 
            poison_ratio, 
            poison_type, 
            trigger_img, 
            trigger_size,
            random_loc=False, 
            mask=None, 
            poison=None
        ):

        '''
        Args:
            dataset (torch.utils.data.Dataset): Dataset to be poisoned
            target_class (int): Target class for the attack
            poison_ratio (float): Ratio of poisoned samples in the dataset
            poison_type (str): Type of poison. Can be 'clean' or 'dirty'
            trigger_img (str): Name of the trigger image to use
            trigger_size (int): Size of the trigger image
            mask (torch.Tensor): Mask to apply to trigger, if already available
            poison (torch.Tensor): Trigger to apply to samples, if already available
        '''

        self.trigger_img = trigger_img
        self.trigger_size = trigger_size
        self.random_loc = random_loc

        super().__init__(dataset, poison_ratio, poison_type, target_class, mask, poison)

        # load trigger image and transform it to a tensor
        trigger_img_path = os.path.join(os.path.dirname(__file__), 'triggers', 'badnets' , self.trigger_img + '.png')
        trigger = Image.open(trigger_img_path).resize((self.trigger_size, self.trigger_size))
        self.trigger = transforms.ToTensor()(trigger)

        # If random_loc is False, then fix the location of the trigger
        if not random_loc:
            
            _ , img_height, img_width = self.sample_shape
            _ , trigger_height, trigger_width = self.trigger.shape

            # pass the position of the trigger to bottom right corner
            x_start_pos = img_width - trigger_width - 1
            y_start_pos = img_height - trigger_height - 1

            self.mask, self.poison = self.get_poison(x_start_pos, y_start_pos)


    def __getitem__(self, index):

        '''
            Method to get a sample from the dataset.

            Args:
                index (int): Index of the sample to get

            Returns:
                tuple: (poisoned_sample, label, poisoned_label)
        '''

        sample, label = self.dataset[index]

        if index not in self.poisoned_indices:
            return sample, label, -1
            
        if self.random_loc:
            mask, poison = self.get_poison()
            poisoned_sample, poisoned_label = self.poison_sample(sample, label, mask, poison)
        else:
            poisoned_sample, poisoned_label = self.poison_sample(sample, label)

        return poisoned_sample, label, poisoned_label

    
    def get_poison(self, x_start_pos=None, y_start_pos=None):

        '''
            Method to get the poison and its mask for BadNets.

            Returns:
                mask (torch.Tensor): Mask to apply to trigger
                poison (torch.Tensor): Trigger to apply to samples
        '''

        # Get the shape of the image and the trigger
        img_channels, img_height, img_width = self.sample_shape
        trigger_channels, trigger_height, trigger_width = self.trigger.shape

        # Check channels consistency of the trigger and the image and fix it if necessary
        if trigger_channels > img_channels:
            # convert rgb to grayscale
            poison = self.trigger.mean(0, keepdim=True)
        else:
            poison = self.trigger
        
        if not x_start_pos and not y_start_pos:
            # Create a random position for the trigger
            x_start_pos = random.randint(0, img_width - trigger_width - 1)
            y_start_pos = random.randint(0, img_height - trigger_height - 1)

        # create a mask of the same size as the image
        mask = torch.ones((img_channels, img_height, img_width))
        mask[:, y_start_pos:y_start_pos+trigger_height, x_start_pos:x_start_pos+trigger_width] = 0

        # creating trigger mask to expand poison to the size of the image
        trigger_mask = torch.zeros((img_channels, img_height, img_width))
        # trigger_mask[:, y_start_pos:y_start_pos+trigger_height, x_start_pos:x_start_pos+trigger_width] = poison
        trigger_mask[:, y_start_pos:y_start_pos+trigger_height, x_start_pos:x_start_pos+trigger_width] = poison

        # poison is a mask with 1 everywhere except where the trigger is
        poison = trigger_mask

        return mask, poison


    # returns a poisoned dataset instance with the same parameters as the current instance
    # useful to poison test dataset with the same parameters as the train dataset
    def poison_transform(self, dataset, poison_ratio):

        '''
            Method to return a poisoned BadNets dataset with the same parameters as the current instance.

            Args:
                dataset (torch.utils.data.Dataset): Dataset to be poisoned
                poison_ratio (float): Ratio of poisoned samples
            
            Returns:
                BadNetsPoison: Poisoned dataset
        '''

        return type(self)(
            dataset=dataset,
            target_class=self.target_class,
            poison_ratio=poison_ratio, 
            poison_type=self.poison_type, 
            trigger_img=self.trigger_img, 
            trigger_size=self.trigger_size,
            random_loc=self.random_loc,
            mask=self.mask,
            poison=self.poison
        )
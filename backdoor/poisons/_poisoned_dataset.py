from abc import abstractmethod
import logging

import random
import numpy as np
import torch
from torchvision import transforms

class PoisonedDataset(torch.utils.data.Dataset):


    def __init__(self, dataset, poison_type, poison_ratio, target_class, mask, poison, log_file=None, seed=None):
        
        # set seed if provided
        if seed:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

        # initialize variables
        self.dataset = dataset
        self.poison_type = poison_type
        self.poison_ratio = poison_ratio
        self.target_class = target_class

        self.sample_shape = transforms.ToTensor()(dataset[0][0]).shape
        
        # track original labels for future use
        self.original_labels = dataset.targets
        # get number of classes
        self.num_classes = len(set(self.original_labels))

        # get indices of poisoned samples
        if self.poison_type == 'dirty':        
            self.num_poisoned_samples = int(self.poison_ratio * len(dataset))
            # replace=False ensures no duplicates
            self.poisoned_indices = set(np.random.choice(range(len(dataset)), self.num_poisoned_samples, replace=False)) 
        
        elif self.poison_type == 'clean':    
            target_indices = np.where(self.original_labels == self.target_class)[0] # indices of target_class only
            self.num_poisoned_samples = int(self.poison_ratio * len(target_indices))
            self.poisoned_indices = set(np.random.choice(target_indices, self.num_poisoned_samples))

        # check if mask and poison are provided
        if mask is None or poison is None:
            self.mask, self.poison = self.get_poison()
        else:
            self.mask = mask
            self.poison = poison

        self.logger = self.setup_logger(log_file)

    # function to get the poison for different kind of poisoning techniques
    @abstractmethod
    def get_poison(self):
        return NotImplementedError("get_poison() is an abstract method and needs to be implemented in the child class")
    

    # caannot use this function before poisoning the dataset because the poisoned dataset is not yet created
    def __getitem__(self, index):

        # (TODO) check if the index is tensor !!! Took two hours for me to figure it out T_T

        sample, label = self.dataset[index]
        poisoned_label = -1
        # check if the sample is poisoned
        if index in self.poisoned_indices:
            sample, poisoned_label = self.poison_sample(sample, label)
        # helps to keep track of the original labels
        return sample, label, poisoned_label


    # similarly cannot use this function as well
    def __len__(self):
        # return length of the (poisoned) dataset
        return len(self.dataset)


    # function to get the poisoned dataset if user wishes
    @property
    def poisoned_dataset(self):
        return self.poison_transform(self.dataset, self.poison_ratio)

    
    # function to get the original targets of the dataset
    @property
    def original_targets(self):
        return self.dataset.targets

    # function to get the targets of the poisoned dataset
    @property
    def targets(self):
        # first get the original targets
        targets = self.dataset.targets
        # clone the targets
        poisoned_targets = targets.clone()
        # change the targets of poisoned samples
        for idx in self.poisoned_indices:
            poisoned_targets[idx] = self.poison_label(targets[idx])
        # iterate through the dataset and get poisoned labels as a tensor
        return poisoned_targets


    # Get poisoned label based on the type of poisoning
    def poison_label(self, label):

        # -1 corresponds to all classes
        if self.target_class == -1:
            poisoned_label = (label + 1) % self.num_classes
        else:
            poisoned_label = self.target_class

        return poisoned_label


    # Poison a single sample
    def poison_sample(self, sample, label):

        poisoned_sample = sample.clone()

        # apply mask
        poisoned_sample = sample * self.mask

        # add poison
        poisoned_sample.add_(self.poison)

        return poisoned_sample, self.poison_label(label)
    

    def setup_logger(self, log_file):
        # Setup logger
        logger = logging.getLogger(__name__)

        # Setup console handler with a DEBUG log level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        logger.addHandler(console_handler)

        # Setup file handler with a INFO log level
        if log_file is not None:
            
            # Check if directory exists
            from os.path import dirname, exists, realpath
            # if not exists(dirname(log_file)):
            #     print(dirname(log_file))
            #     # print(realpath(log_file))
            #     raise ValueError('Directory does not exist: {}'.format(dirname(log_file)))

            file_handler = logging.FileHandler(log_file, 'w')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(file_handler)

        # Set logger level to DEBUG
        logger.setLevel(logging.DEBUG)

        return logger
    

# Dataset that only returns the labels
class Labels(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __getitem__(self, index):
        return self.dataset[index][1]
    
    def __len__(self):
        return len(self.dataset)
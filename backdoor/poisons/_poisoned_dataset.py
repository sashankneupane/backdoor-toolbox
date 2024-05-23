from abc import abstractmethod
import logging

import random
import numpy as np
import torch
from torchvision import transforms

class PoisonedDataset(torch.utils.data.Dataset):


    def __init__(self, dataset, poison_ratio, poison_type, target_class, mask=None, poison=None, log_file=None, seed=None):

        '''
            Base class for poisoned Datasets for all the attacks implemented in this repo.

            Args:
                dataset (torch.utils.data.Dataset): Dataset to poison. Can be both train or test.
                poison_ratio (float): Ratio of samples to poison. 0 <= poison_ratio <= 1.
                poison_type (str): Type of poisoning to perform. Can be 'clean' or 'dirty'.
                target_class (int): Class to target for poisoning.
                mask (torch.Tensor): Mask to apply to trigger.
                poison (torch.Tensor): Trigger to apply to samples.
                log_file (str): Path to log file.
                seed (int): Seed for randomness in the experiments and attacks.
        '''
        
        # set seed if provided
        if seed:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

        # initialize variables
        self.dataset = dataset
        self.poison_ratio = poison_ratio
        self.poison_type = poison_type
        self.target_class = target_class
        self.mask = mask
        self.poison = poison

        self.original_labels = Labels(dataset)

        self.sample_shape = dataset[0][0].shape
        
        # get number of classes
        if isinstance(self.dataset, torch.utils.data.Subset):
            self.num_classes = len(set(self.dataset.dataset.classes))
        else:
            self.num_classes = len(set(self.dataset.classes))

        # get indices of poisoned samples
        if self.poison_type == 'dirty':
            self.num_poisoned_samples = int(self.poison_ratio * len(dataset))
            # replace=False ensures no duplicates
            self.poisoned_indices = set(np.random.choice(range(len(dataset)), self.num_poisoned_samples, replace=False)) 
        
        elif self.poison_type == 'clean':    
            target_indices = np.where(self.original_labels == self.target_class)[0] # indices of target_class only
            self.num_poisoned_samples = int(self.poison_ratio * len(target_indices))
            self.poisoned_indices = set(np.random.choice(target_indices, self.num_poisoned_samples))

        # Setup logger
        self.logger = self.setup_logger(log_file)

        # Validate arguments
        self.validate()


    def validate(self):
        
        '''
            Method that validates the arguments passed to the constructor.
        '''

        # check if dataset is an instance of accepted classes
        # ImageFolder and Subset are subclasses of torch.utils.data.Dataset so they are accepted
        if not isinstance(self.dataset, torch.utils.data.Dataset):
            raise ValueError('Dataset must be an instance of torch.utils.data.Dataset.')

        # poison ratio must be floats between 0 and 1. check if they can be converted to floats
        self.poison_ratio = float(self.poison_ratio)
        if self.poison_ratio < 0 or self.poison_ratio > 1:
            raise ValueError('Poison ratio must be between 0 and 1.')
        
        # poison type must be either 'clean' or 'dirty'
        if self.poison_type not in ['clean', 'dirty']:
            raise ValueError('Poison type must be either \'clean\' or \'dirty\'.')
        
        # target class must be between 0 and num_classes - 1
        self.target_class = int(self.target_class)
        if self.target_class != -1 and (self.target_class < 0 or self.target_class > self.num_classes - 1):
            raise ValueError('Target class must be between 0 and num_classes - 1.')

        if self.poison_type == 'clean' and (self.target_class < 0 and self.target_class > self.num_classes - 1):
            raise ValueError('Target class must be specified when poisoning clean.')


    @abstractmethod
    def get_poison(self):
        '''
            Abstract method to get the poison for different kind of poisoning techniques.
        '''
        return NotImplementedError("get_poison() is an abstract method and needs to be implemented in the child class.")
    

    def __getitem__(self, index):

        '''
            Method to get a sample from the dataset. Cannot use this before poisoning the dataset because poisoned_dataset is not yet created.

            Args:
                index (int): Index of the sample to get.
            
            Returns:
                tuple: (sample, label, poisoned_label)
        '''

        sample, label = self.dataset[index]
        poisoned_label = -1

        # Poison the label if the sample is poisoned
        if index in self.poisoned_indices:
            poisoned_label = self.poison_label(label)
        
        # Return the clean sample, label and the poisoned label
        return sample, label, poisoned_label



    def __len__(self):
        '''
            Method to get the length of the dataset. Cannot use this before poisoning the dataset because poisoned_dataset is not yet created.

            Returns:
                int: Length of the dataset.
        '''
        return len(self.dataset)



    @property
    def poisoned_dataset(self):
        '''
            Property to get the poisoned dataset.
        '''
        return self.poison_transform(self.dataset, self.poison_ratio)

    

    @property
    def original_targets(self):
        '''
            Property to get the original targets of the dataset.

            Returns:
                torch.Tensor: Original targets of the dataset.
        '''
        return self.dataset.targets


    @property
    def poisoned_targets(self):
        '''
            Property to get the targets of the poisoned dataset.

            Returns:
                torch.Tensor: Targets of the poisoned dataset.
        '''

        # get the targets of the dataset
        targets = self.dataset.targets
        # clone the targets
        poisoned_targets = targets.clone()
        # change the targets of poisoned samples
        for idx in self.poisoned_indices:
            poisoned_targets[idx] = self.poison_label(targets[idx])
        # iterate through the dataset and get poisoned labels as a tensor
        
        return poisoned_targets



    def poison_label(self, label):
        '''
            Method to poison the label based on the target class and poison type.

            Args:
                label (int): Label before poison transformation.

            Returns:
                int: Label after poison transformation.
        '''

        # -1 corresponds to all classes
        if self.target_class == -1:
            poisoned_label = (label + 1) % self.num_classes
        else:
            poisoned_label = self.target_class

        return poisoned_label



    def poison_sample(self, sample, label, mask=None, poison=None):
        '''
            Method to poison a sample.

            Args:
                sample (torch.Tensor): Sample to be poisoned.
                label (int): Label of the sample to be poisoned.
                mask (torch.Tensor): Mask to be applied to the sample. Defaults to self.mask
                poison (torch.Tensor): Poison to be added to the sample. Defaults to self.poison 
            Returns:
                tuple: (poisoned_sample, poisoned_label)
        '''

        # Default values for mask and poison if not provided
        if mask is None:
            mask = self.mask
        if poison is None:
            poison = self.poison

        poisoned_sample = sample.clone()

        # apply mask
        poisoned_sample = sample * mask

        # add poison
        poisoned_sample.add_(poison)

        return poisoned_sample, self.poison_label(label)
    

    def setup_logger(self, log_file):
        '''
            Method to setup the logger.

            Args:
                log_file (str): Path to the log file.

            Returns:
                logging.Logger: Logger object.
        '''

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
    


class Labels(torch.utils.data.Dataset):
    '''
        Class to get the labels of a dataset.
    '''
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __getitem__(self, index):
        return self.dataset[index][1]
    
    def __len__(self):
        return len(self.dataset)
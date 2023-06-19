from abc import abstractmethod

import numpy as np
import torch

class PoisonedDataset(torch.utils.data.Dataset):


    def __init__(self, dataset, poison_type, poison_ratio, target_class, mask, poison):

        # initialize variables
        self.dataset = dataset
        self.poison_type = poison_type
        self.poison_ratio = poison_ratio
        self.target_class = target_class

        self.sample_shape = dataset[0][0].shape
        
        # track original labels for future use
        self.original_labels = dataset.targets
        # get number of classes
        self.num_classes = len(set(self.original_labels))

        # get indices of poisoned samples
        if self.poison_type == 'dirty':        
            self.num_poisoned_samples = int(self.poison_ratio * len(dataset))
            self.poisoned_indices = set(np.random.choice(range(len(dataset)), self.num_poisoned_samples))
        
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


    # function to get poisoned label based on the type of poisoning
    def poison_label(self, label):

        # -1 corresponds to all classes
        if self.target_class == -1:
            poisoned_label = (label + 1) % self.num_classes
        else:
            poisoned_label = self.target_class

        return poisoned_label


    # function to poison a single sample
    def poison_sample(self, sample, label):

        poisoned_sample = sample.clone()

        # apply mask
        poisoned_sample = sample * self.mask

        # add poison
        # poisoned_sample = torch.clamp(poisoned_sample + self.poison, 0, 1)
        poisoned_sample.add_(self.poison)

        return poisoned_sample, self.poison_label(label)
    

    # returns a poisoned dataset instance with the same parameters as the current instance
    # useful to poison test dataset with the same parameters as the train dataset
    def poison_transform(self, dataset, poison_ratio):

        return type(self)(
            dataset,
            poison_ratio=poison_ratio, 
            target_class=self.target_class, 
            poison_type=self.poison_type, 
            trigger_img=self.trigger_img, 
            trigger_size=self.trigger_size, 
            mask=self.mask, 
            poison=self.poison
        )   


"""
Hidden Trigger Backdoor Attacks
A. Saha, A. Subramanya, H. Pirsiavash
arXiv:1910.00033 (2019)
"""

import os
import sys
import time
from PIL import Image
import cv2

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

from ._poisoned_dataset import PoisonedDataset
from ..utils import AverageMeter


# Dataset that gets samples from only one class
class HTBADataset(data.Dataset):

    def __init__(self, dataset, label):
        
        self.dataset = dataset
        self.label = label

        self.targets = dataset.targets
        self.classes = dataset.classes

        self.indices = torch.where(torch.tensor(self.targets) == self.label)[0]

        # use indexing to get self.targets
        self.targets = [self.targets[i] for i in self.indices]
        self.classes = list(set(self.targets))

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]
    
    def __len__(self):
        return len(self.indices)


class HTBAPoison(PoisonedDataset):

    def __init__(
            self,
            device,
            dataset,
            pretrained_model,
            source_class,
            target_class,
            split_info,
            poison_type='dirty',
            trigger_img='trigger_10.png',
            poison_ratio=1.0,
            trigger_size=10,
            epochs=10,
            eps=0.1,
            lr=0.1,
            random_loc=False,
            log_file=None,
            seed=None
    ) -> None:
        
        shape = transforms.ToTensor()(dataset[0][0]).shape

        mask = torch.ones(shape)
        poison = torch.zeros(shape)

        super().__init__(dataset, poison_type, poison_ratio, target_class, mask, poison, log_file, seed)
                
        self.device = device
        self.logger.info('Device: {}'.format(self.device))

        self.poison_num, self.binary_train_num, self.test_num = split_info

        self.pretrained_model = pretrained_model
        self.logger.info('Pretrained model: {}'.format(self.pretrained_model))

        self.source_class = source_class
        # get dataset containing only source and target class
        self.source_dataset = HTBADataset(dataset, self.source_class)
        self.target_dataset = HTBADataset(dataset, self.target_class)

        # get poisoned dataset
        self.poisoned_indices = np.random.choice(len(self.source_dataset), self.poison_num, replace=False)
        self.poison_dataset = data.Subset(self.source_dataset, self.poisoned_indices)

        trigger_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'triggers', 'htba', trigger_img)
        self.trigger = Image.open(trigger_path).convert('RGB')
        self.trigger = transforms.Resize((trigger_size, trigger_size))(self.trigger)

        self.trigger_size = trigger_size
        self.epochs = epochs
        self.eps = eps
        self.lr = lr
        self.random_loc = random_loc
        
    
    def generate_poison(self, batch_size, num_workers):
        
        for epoch in range(self.epochs):

            start = time.time()
            losses = AverageMeter()


            trigger = trigger.unsqueeze(0).to(self.device)

            self.target_dataset = None
            self.source_dataset = None

            train_target_loader = torch.utils.data.DataLoader(
                self.target_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
            )
            train_source_loader = torch.utils.data.DataLoader(
                self.source_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
            )

            iter_target = iter(train_target_loader)
            iter_source = iter(train_source_loader)

            # Iterators
            for i in range(len(train_target_loader)):

                # load one batch of source and one batch of target
                (source_samples, source_labels) = next(iter_source)
                (target_samples, target_labels) = next(iter_target)

                # move to device
                source_samples = source_samples.to(self.device)
                source_labels = source_labels.to(self.device)
                target_samples = target_samples.to(self.device)
                target_labels = target_labels.to(self.device)

                perturbation = nn.Parameter(torch.zeros(target_samples.shape, requires_grad=True).to(self.device))

                for j in range(self.source_samples.size(0)):

                    if not self.random_loc:
                        # set trigger in bottom right corner
                        start_x = self.source_samples.size(2) - self.trigger_size
                        start_y = self.source_samples.size(3) - self.trigger_size
                    else:
                        start_x = torch.randint(0, self.source_samples.size(2) - self.trigger_size, (1,))
                        start_y = torch.randint(0, self.source_samples.size(3) - self.trigger_size, (1,))

                    # paste trigger on source images
                    source_samples[j, :, start_x:start_x+self.trigger_size, start_y:start_y+self.trigger_size] = trigger
           
                
    def __getitem__(self, index):
        return super().__getitem__(index)
    

    def save_poisoned_dataset(self, path):
        
        self.logger.info('Saving poisoned dataset to {}'.format(path))
        torch.save(self.poison_dataset, path)


    def poison_sample(self, sample, label):

        # get random location for trigger
        start_x = torch.randint(0, sample.size(1) - self.trigger_size, (1,))
        start_y = torch.randint(0, sample.size(2) - self.trigger_size, (1,))

        # paste trigger on source images
        sample[:, start_x:start_x+self.trigger_size, start_y:start_y+self.trigger_size] = self.trigger

        return sample, label


    def get_poison(self):
        
        # get random sample from poisoned dataset
        sample, label = self.poison_dataset[np.random.randint(len(self.poison_dataset))]
        return self.poison_sample(sample, label)


    def poison_transform(self, dataset):
        
        # return the same instance with a different dataset
        return HTBAPoison(
            self.device,
            dataset,
            self.pretrained_model,
            self.source_class,
            self.target_class,
            self.split_info,
            self.poison_type,
            self.trigger_img,
            self.poison_ratio,
            self.trigger_size,
            self.epochs,
            self.eps,
            self.lr,
            self.random_loc,
            self.log_file,
            self.seed
        )
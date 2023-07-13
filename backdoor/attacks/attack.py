import random
from abc import ABC, abstractmethod

import torch
import numpy as np

class Attack(ABC):

    def __init__(
        self,
        device,
        classifier,
        trainset,
        testset,
        batch_size,
        target_class,
        seed=0
    ) -> None:
        
        # set seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        # training paramters
        self.device = device
        self.classifier = classifier
        self.trainset = trainset
        self.testset = testset
        self.batch_size = batch_size
        self.target_class = target_class


    @abstractmethod
    def attack(self):
        raise NotImplementedError("Attack is an abstract class. Please implement the train method.")
    
    @abstractmethod
    def evaluate_attack(self):
        raise NotImplementedError("Attack is an abstract class. Please implement the evaluate_attack method.")

    def save_model(self, path):
        torch.save(self.classifier.state_dict(), path)
        print(f"\nModel saved to {path}")
from abc import ABC, abstractmethod

import torch

class Attack(ABC):

    def __init__(
        self,
        device,
        model,
        trainset,
        testset,
        epochs,
        batch_size,
        optimizer,
        loss_function
    ) -> None:
        
        # training paramters
        self.device = device
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss_function = loss_function


    @abstractmethod
    def attack(self):
        raise NotImplementedError("Attack is an abstract class. Please implement the train method.")
    
    @abstractmethod
    def evaluate_attack(self):
        raise NotImplementedError("Attack is an abstract class. Please implement the evaluate_attack method.")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
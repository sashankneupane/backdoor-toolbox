import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from .attack import Attack


class NarcissusAttack(Attack):

    def __init__(self, device, model, surrogate_model, trainset, testset, pood_trainset, epochs, batch_size, optimizer, loss_function, attack_args, seed=0):
        
        super().__init__(device, model, trainset, testset, epochs, batch_size, optimizer, loss_function, seed)
        

    def attack(self):
        pass

    def evaluate_attack(self):
        pass
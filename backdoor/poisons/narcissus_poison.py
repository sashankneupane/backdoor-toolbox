from copy import deepcopy
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset

from ._poisoned_dataset import PoisonedDataset


# Dataset that concatinates the pood dataset and the target class dataset
class NarcissusDataset(Dataset):

    def __init__(self, pood_dataset, targetclass_dataset):
        self.pood = pood_dataset
        self.target = targetclass_dataset

    def __getitem__(self, index):
        if index < len(self.pood):
            return self.pood[index]
        else:
            # change the label to a new label for the newly introduced target class data
            img = self.target[index - len(self.pood)][0]
            label = len(self.pood.classes)
            return (img, label)
    
    def __len__(self):
        return len(self.pood) + len(self.target)
    

class FinetuneDataset(Dataset):

    def __init__(self, dataset, label):
        self.dataset = dataset
        self.label = label
    
    def __getitem__(self, index):
        img = self.dataset[index][0]
        return (img, self.label)

    def __len__(self):
        return len(self.dataset)


class NarcissusPoison(PoisonedDataset):

    def __init__(self, device, pood_trainset, target_dataset, surrogate_model, trigger_model, target_class=0):
        
        self.device = device

        self.pood_trainset = pood_trainset
        # creating dataset with only the target class

        self.target_class = target_class
        self.target_indices = np.where(np.array(target_dataset.targets) == target_class)[0]
        self.target_dataset = Subset(target_dataset, self.target_indices)

        # For all the target class samples, returns len(pood_trainset.classes) as the label to train the surrogate model and generate trigger
        self.modified_target_dataset = FinetuneDataset(self.target_dataset, len(pood_trainset.classes))

        self.surrogate_model = surrogate_model
        self.trigger_model = trigger_model

        batch_size = 350

        # Get concatenated dataset with the target class
        self.attack_dataset = NarcissusDataset(self.pood_trainset, self.target_dataset)

        self.surrogate_loader = torch.utils.data.DataLoader(self.attack_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
        self.poi_warm_up_loader = torch.utils.data.DataLoader(self.modified_target_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
        self.trigger_gen_loaders = torch.utils.data.DataLoader(self.modified_target_dataset, batch_size=batch_size, shuffle=True, num_workers=16)


    # Loads the surrogate model if it is already trained
    def load_surrogate(self, surrogate_model):
        self.surrogate_model = surrogate_model

    def load_warmup(self, warmup_model):
        self.warmup_model = warmup_model

    # Train the surrogate model on POOD dataset
    def train_surrogate(self, sur_epochs, criterion, surrogate_opt, surrogate_scheduler):

        loss_list = []

        surrogate_optimizer = surrogate_opt(self.surrogate_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        surrogate_scheduler = surrogate_scheduler(surrogate_optimizer, T_max=sur_epochs)
        
        for epoch in range(0, sur_epochs):
            self.surrogate_model.train()
            loss_list = []
            for images, labels in self.surrogate_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.surrogate_model.zero_grad()
                outputs = self.surrogate_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                loss_list.append(loss.item())
                surrogate_optimizer.step()
            surrogate_scheduler.step()
            avg_loss = sum(loss_list) / len(loss_list)
            print(f'Epoch: {epoch} \tLoss: {avg_loss}')
        
        torch.save(self.surrogate_model.state_dict(), 'surrogate_model.pth')


    # Warmup the surrogate model on the target class dataset
    def poi_warmup(self, warmup_epochs, criterion, warmup_opt):
        
        self.warmup_model = self.trigger_model

        self.warmup_model.train()
        for param in self.warmup_model.parameters():
            param.requires_grad = True

        self.warmup_model.load_state_dict(self.surrogate_model.state_dict())
        self.warmup_optimizer = warmup_opt(self.warmup_model.parameters(), lr=0.1)
        self.warmup_model.to(self.device)

        for epoch in range(warmup_epochs):
            self.warmup_model.train()
            loss_list = []
            for images, labels in self.poi_warm_up_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.warmup_model.zero_grad()
                outputs = self.warmup_model(images)
                loss = criterion(outputs, labels)
                loss.backward(retain_graph=True)
                loss_list.append(float(loss.data))
                self.warmup_optimizer.step()

            ave_loss = np.average(np.array(loss_list))
            print('Epoch:%d, Loss: %e' % (epoch, ave_loss))
        
        torch.save(self.warmup_model.state_dict(), 'warmup_model.pth')

    
    # Generate Trigger
    def generate_trigger(self, trigger_gen_epochs, criterion, optimizer, lr_inf_r=16/255, lr_inf_r_step=0.01):

        trigger_gen_loaders = self.trigger_gen_loaders

        for param in self.warmup_model.parameters():
            param.requires_grad = False
        
        dataset_shape = self.target_dataset[0][0].shape
        noise = torch.zeros((1, *dataset_shape))
        noise = noise.to(self.device)
        
        noise = torch.autograd.Variable(noise, requires_grad=True)

        optimizer = optimizer([noise], lr=lr_inf_r_step)

        for round in range(trigger_gen_epochs):
            loss_list = []
            for images, labels in trigger_gen_loaders:
                
                images, labels = images.to(self.device), labels.to(self.device)
                new_images = images.clone()
                clamped_noise = torch.clamp(noise, -lr_inf_r*2, lr_inf_r*2)
                new_images += clamped_noise
                new_images = torch.clamp(new_images, -1, 1)
                
                per_logits = self.warmup_model(new_images)
                
                loss = criterion(per_logits, labels)
                loss = torch.mean(loss)
                loss_list.append(float(loss.data))
                loss.backward(retain_graph=True)

                optimizer.zero_grad()
                optimizer.step()

                

            avg_loss = sum(loss_list) / len(loss_list)
            avg_grad = torch.mean(torch.abs(noise.grad.data))
            print(f'Round: {round} \tLoss: {avg_loss} \tAvg Grad: {avg_grad}')

            if avg_grad == 0:
                break
        
        noise = torch.clamp(noise, -lr_inf_r*2, lr_inf_r*2)
        noise = noise.detach().cpu().numpy()

        self.noise = noise
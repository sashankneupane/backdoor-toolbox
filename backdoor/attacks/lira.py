import time
from math import ceil, floor
from copy import deepcopy

import warnings
warnings.filterwarnings("ignore")

import torch
from torch import nn
import torchvision.transforms as transforms

from .attack import Attack
from ..poisons import LiraPoison


# Model pipeline to handle all forward passes for the LIRA attack
class LiraModel(nn.Module):

    def __init__(self, trigger_model, fixed_trigger_model, classifier, eps, min_val=None, max_val=None):
        
        super().__init__()

        # running trigger model
        self.trigger_model = trigger_model
        # fixed trigger model
        self.fixed_trigger_model = fixed_trigger_model

        # freeze the fixed trigger_model
        self.freeze_trigger_model()

        # classifier model
        self.classifier = classifier

        self.eps = eps

        self.min_val = min_val
        self.max_val = max_val

    def clip_image(self, x):
        return torch.clamp(x, self.min_val, self.max_val)

    def freeze_trigger_model(self): 
        for param in self.fixed_trigger_model.parameters():
            param.requires_grad = False

    def forward(self, x, poison=False, update='classifier', eps=None):

        if eps is None:
            eps = self.eps

        if poison:
            if update == 'trigger':
                x = x + self.trigger_model(x) * self.eps
            elif update == 'classifier':
                x = x + self.fixed_trigger_model(x) * self.eps
            else:
                raise ValueError('update must be either trigger or classifier')
        
        x = self.clip_image(x)
        x = self.classifier(x)

        return x



class LIRA(Attack):

    '''
    
    Implementation of the Lira attack from the paper:
    "LIRA: Learnable, Imperceptible, and Robust backdoor Attacks"

    Official Implementation:
    https://github.com/sunbelbd/invisible_backdoor_attacks

    '''

    def __init__(
            self, 
            device, # device to run the attack on
            classifier, # victim model
            trigger_model, # trigger generator model
            trainset, # clean training set
            testset, # clean testing set
            target_class, # target class to be misclassified into
            batch_size, # batch size for training
            num_workers=8, # number of workers for dataloader
            logfile=None, # path to the logfile
            seed=None, # random seed
        ) -> None:

        super().__init__(
            device=device, 
            classifier=classifier, 
            trainset=trainset, 
            testset=testset, 
            batch_size=batch_size, 
            target_class=target_class,
            logfile=logfile,
            seed=seed
            )

        # Running trigger model
        self.trigger_model = trigger_model.to(self.device)

        # Create a copy of the trigger model that gets fixed for classifier loss
        self.fixed_trigger_model = deepcopy(self.trigger_model).to(self.device)

        # Create a clean test loader for testing the poisoned model
        self.num_workers = num_workers
        self.clean_testloader = torch.utils.data.DataLoader(
            self.testset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )

        # set min and max values for the dataset
        self.set_minmax(self.trainset)

        # flag to keep track of whether the attack has been run
        self.attacked = False

    
    def set_minmax(self, dataset):

        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )
    
        min_val = float('inf')
        max_val = float('-inf')

        for samples, _ in dataloader:
            min_val = min(min_val, samples.min())
            max_val = max(max_val, samples.max())

        # round up for max_val and round down the min_val if min_val is negative
        max_val = ceil(max_val)
        min_val = floor(min_val)
        
        self.min_val = torch.tensor(min_val, device=self.device)
        self.max_val = torch.tensor(max_val, device=self.device)

        self.logger.debug(f'\n\n\nmin_val: {self.min_val}, max_val: {self.max_val}\n\n\n')


    def apply_trigger(self, samples, trigger=None, eps=None):

        '''
            This method applies trigger to a sample or a batch of samples.

            Args:
                samples(tensor): a sample or a batch of samples
                trigger(nn.Module): trigger model to be applied. If None, the trigger model of current instance is used.
                eps(int): epsilon value for stealthiness. If None, the epsilon value of current instance is used.

            Returns:
                poisoned_samples(tensor): a sample or a batch of samples with trigger applied 
        '''

        if trigger is None:
            trigger = self.trigger_model
        
        if eps is None:
            eps = self.eps

        # if the samples are a batch
        if len(samples.shape) == 4:
            poisoned_samples = samples + eps * trigger(samples)
        elif len(samples.shape) == 3:
            poisoned_samples = samples + eps * trigger(samples.unsqueeze(0)).squeeze(0)
        else:
            raise RuntimeError('Invalid shape of samples. Expected 3 or 4 dimensions.')

        # Clip the poisoned samples to be in the range [-1, 1]        
        return poisoned_samples
    


    def get_poisoned_trainset(self):

        '''
            This method returns the poisoned train set with the trigger model applied to it.

            Args:
                None
            
            Returns:
                LiraPoison instance with the poisoned training set applied with the trigger model.
        '''

        if self.attacked:
            return LiraPoison(self.trainset, self.target_class, self.fixed_trigger_model, self.eps)
        else:
            # Raise error if the attack has not been run and there is no trigger model
            return RuntimeError('Please run the attack first. Poisoned dataset instance without trigger model is returned.')



    def get_poisoned_testset(self):
        
        '''
            This method returns the poisoned test set with the trigger model applied to it.

            Args:
                None

            Returns:
                LiraPoison instance with the poisoned test set applied with the trigger model.
        '''

        if self.attacked:
            return LiraPoison(self.testset, self.target_class, self.fixed_trigger_model, self.eps)
        else:
            # Raise error if the attack has not been run and there is no trigger model
            return RuntimeError('Please run the attack first. No trigger model found to poison the testset.')
    
    

    # Perform LIRA attack on the trainset
    def attack(
            self, 
            epochs, 
            finetune_epochs, 
            optimizer, 
            trigger_optimizer,
            finetune_optimizer,
            finetune_scheduler=None,
            loss_function=nn.CrossEntropyLoss(),
            finetune_loss_function=nn.CrossEntropyLoss(),
            update_trigger_epochs=1, 
            alpha=0.5, 
            finetune_alpha=0.5,
            eps=0.01,
            finetune_test_eps=0.01,
            eval_every=1,
        ) -> None:

        '''
            This method performs the LIRA attack on the trainset.

            Args:
                epochs(int): number of epochs to train the trigger model
                finetune_epochs(int): number of epochs to finetune the classifier
                optimizer(torch.optim): optimizer to train the trigger model
                trigger_optimizer(torch.optim): optimizer to train the classifier
                finetune_optimizer(torch.optim): optimizer to finetune the classifier
                finetune_scheduler(torch.optim.lr_scheduler): scheduler to finetune the classifier
                loss_function(nn.Module): loss function to train the trigger model
                finetune_loss_function(nn.Module): loss function to finetune the classifier
                update_trigger_epochs(int): number of epochs to update the trigger model
                alpha(float): weight of the trigger loss in the total loss during Stage I
                finetune_alpha(float): weight of the classifier loss in the total loss during Stage II
                eps(float): epsilon value for stealthiness during Stage I
                finetune_eps(float): epsilon value for stealthiness during Stage II
                eval_every(int): number of epochs after which the model is evaluated on the test set
        '''

        self.eval_every = eval_every
        
        self.epochs = epochs
        self.optimizer = optimizer
        self.trigger_optimizer = trigger_optimizer
        self.loss_function = loss_function
        self.update_trigger_epochs = update_trigger_epochs
        self.alpha = alpha
        self.eps = eps

        self.finetune_epochs = finetune_epochs
        self.finetune_optimizer = finetune_optimizer
        self.finetune_scheduler = finetune_scheduler
        self.finetune_loss_function = finetune_loss_function
        self.finetune_alpha = finetune_alpha
        self.finetune_test_eps = finetune_test_eps

        
        self.log_attack_info()
        self.logger.info(f'\nBEGIN ATTACK')

        self.attack_model = LiraModel(self.trigger_model, self.fixed_trigger_model, self.classifier, self.eps, self.min_val, self.max_val)

        # Stage I LIRA attack with alternating optimization
        self.logger.info('\nStage I LIRA attack with alternating optimization')

        
        # Track the losses
        self.trainlosses = []
        self.triggerlosses = []
        self.classifierlosses = []

        # self.normalize_transform = self.get_normalization_transform(self.trainset)

        # Get poisoned_trainset and poisoned_trainloader without passing trigger_model
        self.poisoned_trainset = LiraPoison(self.trainset, self.target_class, self.eps)
        self.poisoned_trainloader = torch.utils.data.DataLoader(
            self.poisoned_trainset,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        # Get non-target indices for the testset
        targets = torch.tensor(self.testset.targets)
        self.non_target_indices = torch.where(targets != self.target_class)[0]
        asr_testset = torch.utils.data.Subset(self.testset, self.non_target_indices)
        self.asr_testset = LiraPoison(asr_testset, self.target_class, self.eps)
        self.asr_testloader = torch.utils.data.DataLoader(
            self.asr_testset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        # Iterate through the epochs
        for epoch in range(self.epochs):

            self.classifier.train()
            self.trigger_model.train()

            # Loss list for current epoch
            triggerlosses = []
            classifierlosses = []


            # Iterate through the the trainloader
            for samples, labels, poisoned_labels in self.poisoned_trainloader:
                
                # moving the data to the device
                samples = samples.to(self.device)
                labels = labels.to(self.device)
                poisoned_labels = poisoned_labels.to(self.device)

                # TRIGGER GENERATOR UPDATE --------------------------------------------------------------------
                # ---------------------------------------------------------------------------------------------
                
                poisoned_outputs = self.attack_model(samples, poison=True, update='trigger')
                trigger_loss = self.loss_function(poisoned_outputs, poisoned_labels)

                # This will backpropagate the loss to the classifier as well as the running trigger model
                self.trigger_optimizer.zero_grad()
                trigger_loss.backward()
                self.trigger_optimizer.step()


                # CLASSIFIER UPDATE --------------------------------------------------------------------------
                # ---------------------------------------------------------------------------------------------

                # Calculate the classifier loss based on the fixed trigger model not the trigger model
                outputs = self.attack_model(samples)
                poisoned_outputs = self.attack_model(samples, poison=True, update='classifier')

                clean_loss = self.loss_function(outputs, labels)
                poisoned_loss = self.loss_function(poisoned_outputs, poisoned_labels)
                classifier_loss = clean_loss * self.alpha + poisoned_loss * (1 - self.alpha)

                # Clear the gradients (This will also clear gradients accumulated from trigger model update)
                self.optimizer.zero_grad()

                # This will backpropagate the loss to the classifier as well as the trigger model
                classifier_loss.backward()

                self.optimizer.step()
                # We don't need to care about the gradients backpropagated to the fixed trigger model as we won't touch it anywhere in the attacking pipeline

                # Keep track of the losses
                classifierlosses.append(classifier_loss.item())
                triggerlosses.append(trigger_loss.item())


            # Update the weights of the trigger model after update_trigger_epochs
            if (epoch+1) % self.update_trigger_epochs == 0:

                self.fixed_trigger_model.load_state_dict(self.trigger_model.state_dict())

                # Update the trigger model in the poisoned_trainset
                self.poisoned_trainset.trigger_model = self.fixed_trigger_model

            
            # Average trigger loss and classifier loss for the current epoch
            avg_classifier_loss = sum(classifierlosses) / len(classifierlosses)
            avg_trigger_loss = sum(triggerlosses) / len(triggerlosses)

            # Add the losses to the running losses list
            self.triggerlosses += triggerlosses
            self.classifierlosses += classifierlosses

            self.trainlosses.append(avg_classifier_loss)

            # Evaluate the attack every eval_every epochs
            if (epoch+1) % self.eval_every == 0:
                self.logger.info(f'\nEpoch {epoch+1} Classifier Loss: {avg_classifier_loss} Trigger Loss: {avg_trigger_loss}')
                asr = self.evaluate_attack(self.eps)
                
                # if asr == 1.0 and epoch < self.epochs:
                #     self.logger.info(f'Attack Successful at epoch {epoch} with ASR {asr}, so early stopping Stage I')
                #     break

        # Stage II LIRA attack with backdoor finetuning
        self.logger.info('\nStage II LIRA attack with backdoor finetuning')

        for epoch in range(self.epochs, self.epochs+self.finetune_epochs):
            
            self.attack_model.train()

            finetunelosses = []

            for samples, labels, poisoned_labels in self.poisoned_trainloader:

                samples = samples.to(self.device)
                labels = labels.to(self.device)
                poisoned_labels = poisoned_labels.to(self.device)

                # zero the gradients
                self.optimizer.zero_grad()

                outputs = self.attack_model(samples)
                poisoned_outputs = self.attack_model(samples, poison=True, update='classifier')
                
                clean_loss = self.loss_function(outputs, labels)
                poisoned_loss = self.loss_function(poisoned_outputs, poisoned_labels)
                
                classifier_loss = clean_loss * self.alpha + poisoned_loss * (1 - self.alpha)

                # Calculate the total loss
                classifier_loss.backward()

                # Update the weights of the classifier model
                self.optimizer.step()
                
                finetunelosses.append(classifier_loss.item())
            
            self.classifierlosses += finetunelosses

            avg_finetuneloss = sum(finetunelosses) / len(finetunelosses)
            self.trainlosses.append(avg_finetuneloss)
            
            # Evaluate the attack every eval_every epochs
            if (epoch+1) % self.eval_every == 0:
                self.logger.info(f'\nEpoch {epoch+1} Finetune Loss: {avg_finetuneloss}')
                asr = self.evaluate_attack(self.finetune_test_eps)
            

            # Step the scheduler
            if self.finetune_scheduler is not None:
                self.finetune_scheduler.step()


        # set the attacked flag to true
        self.attacked = True



    def evaluate_attack(self, eps=None):

        '''
            Evaluate the attack with clean test accuracy, target class accuracy, and ASR.

            Args:
                epoch(int): current epoch
                classifier_loss(float): average classifier loss for the current epoch
                trigger_loss(float): average trigger loss for the current epoch
        '''
        
        self.attack_model.eval()

        if eps is None:
            eps = self.eps

        num_classes = self.poisoned_trainset.num_classes

        # tensors to keep track of number of correct predictions per class
        class_correct = torch.zeros(num_classes).to(self.device)
        class_total = torch.zeros(num_classes).to(self.device)

        with torch.no_grad():
            
            for i, (samples, labels) in enumerate(self.clean_testloader):

                samples = samples.to(self.device)
                labels = labels.to(self.device)

                outputs = self.attack_model(samples)

                _, predicted = torch.max(outputs, 1)

                # update the number of correct predictions per class
                for label in range(num_classes):
                    labels_mask = (labels == label)
                    class_total[label] += labels_mask.sum().item()
                    class_correct[label] += (predicted[labels_mask] == label).sum().item()

        # Calculate the clean test accuracy and clean target class accuracy
        clean_test_accuracy = class_correct.sum().item() / class_total.sum().item()
        target_class_accuracy = class_correct[self.target_class].item() / class_total[self.target_class].item()
        
        # Log the accuracies
        self.logger.info(f'Clean test accuracy: {clean_test_accuracy} | Clean target class accuracy: {target_class_accuracy} | Class accuracies: {list((class_correct / class_total).cpu().numpy())}')


        # ----------------------------------------------------------------------------------------------------
        # Calculate the ASR
        # ----------------------------------------------------------------------------------------------------

        attack_success_count = 0

        with torch.no_grad():
            # Iterate over the non-target samples
            for samples, _, poisoned_labels in self.asr_testloader:
                
                samples = samples.to(self.device)
                poisoned_labels = poisoned_labels.to(self.device)

                # Get the outputs of the attack model
                attack_outputs = self.attack_model(samples, poison=True, update='classifier', eps=eps)
                _, attack_predicted = torch.max(attack_outputs, 1)

                # Update the number of correct attack predictions
                attack_success_count += (attack_predicted == poisoned_labels).sum().item()

        attack_success_rate = attack_success_count / len(self.asr_testset)
        self.logger.info(f'Attack Success Rate: {attack_success_rate}')                

        return attack_success_rate


    def log_attack_info(self):
            
        '''
            Log the attack information
        '''

        self.logger.info(f'Attack: LIRA')
        self.logger.info(f'\nClassifier: {self.classifier}')
        self.logger.info(f'\nTrigger Model: {self.trigger_model}')
        self.logger.info(f'\nEpochs: {self.epochs}')
        self.logger.info(f'Finetune Epochs: {self.finetune_epochs}')
        self.logger.info(f'Update Trigger Epochs: {self.update_trigger_epochs}')
        self.logger.info(f'Alpha: {self.alpha}')
        self.logger.info(f'\nOptimizer: {self.optimizer}')
        self.logger.info(f'\nTrigger Optimizer: {self.trigger_optimizer}')
        self.logger.info(f'\nFinetune Optimizer: {self.finetune_optimizer}')
        if self.finetune_scheduler is not None:
            self.logger.info(f'\nFinetune Scheduler: {self.finetune_scheduler}')
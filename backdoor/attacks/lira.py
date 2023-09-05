# from math import ceil, floor
import time
from copy import deepcopy

import torch
from torch import nn
import torchvision.transforms as transforms

from .attack import Attack
from ..poisons import LiraPoison



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

        # dataloader = torch.utils.data.DataLoader(
        #     dataset, 
        #     batch_size=self.batch_size, 
        #     shuffle=False, 
        #     num_workers=self.num_workers
        # )
    
        # min_val = float('inf')
        # max_val = float('-inf')

        # for samples, _ in dataloader:
        #     min_val = min(min_val, samples.min())
        #     max_val = max(max_val, samples.max())
        
        # # self.min_val = torch.tensor(min_val, device=self.device)
        # # self.max_val = torch.tensor(max_val, device=self.device)
         
        self.min_val = -2.1179039301310043
        self.max_val = 2.6399999999999997

        # self.min_val = 0.0
        # self.max_val = 1.0


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
            trigger = self.fixed_trigger_model
        
        if eps is None:
            eps = self.eps

        # check if the samples are a batch of samples or a single sample
        if len(samples.shape) == 4:
            noise = trigger(samples) * eps
        elif len(samples.shape) == 3:
            noise = trigger(samples.unsqueeze(0)).squeeze(0) * eps
        else:
            raise RuntimeError('Invalid shape of samples. Expected 3 or 4 dimensions.')
        
        poisoned_samples = torch.clamp(samples + noise, self.min_val, self.max_val)

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
            eps=0.1,
            finetune_test_eps=0.1,
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

        # self.attack_model = LiraModel(self.trigger_model, self.fixed_trigger_model, self.classifier, self.eps, self.min_val, self.max_val)

        # Stage I LIRA attack with alternating optimization
        self.logger.info('\nStage I LIRA attack with alternating optimization')

        
        # Track the losses
        self.trainlosses = []
        self.triggerlosses = []
        self.cleanlosses = []
        self.poisonedlosses = []
        self.classifierlosses = []



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
            self.fixed_trigger_model.eval()

            # Loss list for current epoch
            triggerlosses = []
            cleanlosses = []
            poisonedlosses = []
            classifierlosses = []


            # Iterate through the the trainloader
            for samples, labels, poisoned_labels in self.poisoned_trainloader:
                
                # moving the data to the device
                samples = samples.to(self.device)
                labels = labels.to(self.device)
                poisoned_labels = poisoned_labels.to(self.device)

                # TRIGGER GENERATOR UPDATE --------------------------------------------------------------------
                # ---------------------------------------------------------------------------------------------

                # Calculate the trigger loss
                poisoned_samples = self.apply_trigger(samples, trigger=self.trigger_model)
                poisoned_outputs = self.classifier(poisoned_samples)
                trigger_loss = self.loss_function(poisoned_outputs, poisoned_labels)
                
                # Update the running trigger model
                self.optimizer.zero_grad()
                self.trigger_optimizer.zero_grad()
                trigger_loss.backward()
                self.trigger_optimizer.step()

                # Add the trigger loss to the list
                triggerlosses.append(trigger_loss.item())


                # CLASSIFIER UPDATE --------------------------------------------------------------------------
                # ---------------------------------------------------------------------------------------------
                poisoned_samples = self.apply_trigger(samples)
                poisoned_outputs = self.classifier(poisoned_samples)
                clean_outputs = self.classifier(samples)
                
                # Calculate the weighted loss and update the classifier
                clean_loss = self.loss_function(clean_outputs, labels)
                poisoned_loss = self.loss_function(poisoned_outputs, poisoned_labels)
                classifier_loss = clean_loss * self.alpha + poisoned_loss * (1 - self.alpha)

                self.optimizer.zero_grad()
                classifier_loss.backward()
                self.optimizer.step()
                
                cleanlosses.append(clean_loss.item())
                poisonedlosses.append(poisoned_loss.item())
                classifierlosses.append(classifier_loss.item())


            
            # Update the weights of the trigger model after update_trigger_epochs
            if (epoch+1) % self.update_trigger_epochs == 0:

                self.fixed_trigger_model.load_state_dict(deepcopy(self.trigger_model.state_dict()))

                # Update the trigger model in the poisoned_trainset
                self.poisoned_trainset.trigger_model = self.fixed_trigger_model

            
            # Average trigger loss and classifier loss for the current epoch
            avg_classifier_loss = sum(classifierlosses) / len(classifierlosses)
            avg_trigger_loss = sum(triggerlosses) / len(triggerlosses)

            # Add the losses to the running losses list
            self.triggerlosses += triggerlosses
            self.cleanlosses += cleanlosses
            self.poisonedlosses += poisonedlosses
            self.classifierlosses += classifierlosses

            self.trainlosses.append(avg_classifier_loss)

            # Evaluate the attack every eval_every epochs and at the last epoch
            if (epoch+1) % self.eval_every == 0 or epoch == self.epochs-1:
                self.logger.info(f'\nEpoch {epoch+1} Classifier Loss: {avg_classifier_loss} Trigger Loss: {avg_trigger_loss}')
                asr = self.evaluate_attack(warmup=True, eps=self.eps)

            # self.save_model('./')

        # Stage II LIRA attack with backdoor finetuning
        self.logger.info('\nStage II LIRA attack with backdoor finetuning')


        for epoch in range(self.epochs, self.epochs+self.finetune_epochs):
            
            # self.attack_model.train()
            self.classifier.train()
            self.trigger_model.eval()
            self.fixed_trigger_model.eval()

            finetunelosses = []

            for samples, labels, poisoned_labels in self.poisoned_trainloader:

                samples = samples.to(self.device)
                labels = labels.to(self.device)
                poisoned_labels = poisoned_labels.to(self.device)

                poisoned_samples = self.apply_trigger(samples, eps=self.finetune_test_eps)

                outputs = self.classifier(samples)
                poisoned_outputs = self.classifier(poisoned_samples)
                
                clean_loss = self.loss_function(outputs, labels)
                poisoned_loss = self.loss_function(poisoned_outputs, poisoned_labels)
                
                classifier_loss = clean_loss * self.finetune_alpha + poisoned_loss * (1 - self.finetune_alpha)

                # Backward pass and update the classifier
                self.finetune_optimizer.zero_grad()
                classifier_loss.backward()
                self.finetune_optimizer.step()
                
                self.cleanlosses.append(clean_loss.item())
                self.poisonedlosses.append(poisoned_loss.item())
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



    def evaluate_attack(self, warmup=False, eps=None):

        '''
            Evaluate the attack with clean test accuracy, target class accuracy, and ASR.

            Args:
                epoch(int): current epoch
                classifier_loss(float): average classifier loss for the current epoch
                trigger_loss(float): average trigger loss for the current epoch
        '''

        # # warm up the classifier
        # if warmup:
        #     test_classifier = deepcopy(self.classifier)
        #     test_classifier.train()
        #     test_optimizer = torch.optim.SGD(test_classifier.parameters(), lr=1e-2, momentum=0.9)

        #     for samples, labels, poisoned_labels in self.poisoned_trainloader:
        #         samples = samples.to(self.device)
        #         labels = labels.to(self.device)
        #         poisoned_labels = poisoned_labels.to(self.device)

        #         outputs = test_classifier(samples)
        #         loss = self.loss_function(outputs, labels)

        #         poisoned_samples = self.apply_trigger(samples)
        #         poisoned_outputs = test_classifier(poisoned_samples)
        #         poisoned_loss = self.loss_function(poisoned_outputs, poisoned_labels)

        #         test_loss = loss * self.alpha + poisoned_loss * (1 - self.alpha)
                
        #         test_optimizer.zero_grad()
        #         test_loss.backward()
        #         test_optimizer.step()
        # else:
        #     test_classifier = self.classifier

        test_classifier = self.classifier

        
        self.classifier.eval()
        self.trigger_model.eval()
        self.fixed_trigger_model.eval()

        if eps is None:
            eps = self.eps

        num_classes = self.poisoned_trainset.num_classes

        # tensors to keep track of number of correct predictions per class
        class_correct = torch.zeros(num_classes).to(self.device)
        class_total = torch.zeros(num_classes).to(self.device)

        with torch.no_grad():
            
            for samples, labels in self.clean_testloader:

                samples = samples.to(self.device)
                labels = labels.to(self.device)

                outputs = test_classifier(samples)
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
                # attack_outputs = self.attack_model(samples, poison=True, update='classifier', eps=eps)
                attack_outputs = test_classifier(self.apply_trigger(samples))
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
        self.logger.info(f'Eps (Stage I): {self.eps}')
        self.logger.info(f'Alpha (Stage I): {self.alpha}')
        self.logger.info(f'Eps (Stage II): {self.finetune_test_eps}')
        self.logger.info(f'Alpha (Stage II): {self.finetune_alpha}')
        self.logger.info(f'\nOptimizer: {self.optimizer}')
        self.logger.info(f'\nTrigger Optimizer: {self.trigger_optimizer}')
        self.logger.info(f'\nFinetune Optimizer: {self.finetune_optimizer}')
        if self.finetune_scheduler is not None:
            self.logger.info(f'\nFinetune Scheduler: {self.finetune_scheduler}')

        
    def save_models(self, path):

        # save both classifier and trigger model in the same file
        torch.save({
            'classifier': self.classifier.state_dict(),
            'trigger_model': self.trigger_model.state_dict(),
        }, path)
from copy import deepcopy

import warnings
warnings.filterwarnings("ignore")

import torch
from torch import nn

from .attack import Attack
from ..poisons import LiraPoison


class LiraModel(nn.Module):

    def __init__(self, trigger_model, fixed_trigger_model, classifier, eps):
        
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

    def freeze_trigger_model(self): 
        for param in self.fixed_trigger_model.parameters():
            param.requires_grad = False

    def forward(self, x, poison=False, update='classifier'):

        if poison:
            if update == 'trigger':
                x = x + self.trigger_model(x) * self.eps
            elif update == 'classifier':
                x = x + self.fixed_trigger_model(x) * self.eps
            else:
                raise ValueError('update must be either trigger or classifier')
            
        x = self.classifier(x)

        return x



class LiraAttack(Attack):

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
            seed=0
        ) -> None:

        super().__init__(
            device=device, 
            classifier=classifier, 
            trainset=trainset, 
            testset=testset, 
            batch_size=batch_size, 
            target_class=target_class, 
            seed=seed
            )

        # Running trigger model
        self.trigger_model = trigger_model

        # Create a copy of the trigger model that gets fixed for classifier loss
        self.fixed_trigger_model = deepcopy(self.trigger_model)

        # flag to keep track of whether the attack has been run
        self.attacked = False



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
            num_workers=8,
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
                num_workers(int): number of workers to use for loading the data
        '''

        self.eval_every = eval_every
        self.num_workers = num_workers

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

        self.attack_model = LiraModel(self.trigger_model, self.fixed_trigger_model, self.classifier, self.eps)

        # Stage I LIRA attack with alternating optimization
        print('\nStage I LIRA attack with alternating optimization')

        
        # Track the losses
        self.trainlosses = []
        self.triggerlosses = []
        self.classifierlosses = []

        # Get poisoned_trainset and poisoned_trainloader without passing trigger_model
        self.poisoned_trainset = LiraPoison(self.trainset, self.target_class, self.eps)
        self.poisoned_trainloader = torch.utils.data.DataLoader(
            self.poisoned_trainset,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        self.poisoned_testset = LiraPoison(self.testset, self.target_class, self.eps)
        self.poisoned_testloader = torch.utils.data.DataLoader(
            self.poisoned_testset,
            self.batch_size,
            shuffle=False,
        )
            
        num_workers=self.num_workers,

        self.trigger_optimizer.zero_grad()
        self.optimizer.zero_grad()

        self.classifier_state_dict = self.classifier.state_dict()

        # Iterate through the epochs
        for epoch in range(self.epochs):

            self.attack_model.train()

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
                
                # we need to update classifier temporarily so we save its state dict and load it later
                self.classifier_state_dict = self.classifier.state_dict()
                
                outputs = self.attack_model(samples)
                poisoned_outputs = self.attack_model(samples, poison=True, update='trigger')

                temp_cleanloss = self.loss_function(outputs, labels)
                temp_poisonedloss = self.loss_function(poisoned_outputs, poisoned_labels)
                temp_loss = temp_cleanloss * self.alpha + temp_poisonedloss * (1 - self.alpha)

                # This will backpropagate the loss to the classifier as well as the running trigger model
                temp_loss.backward()

                # We only want to update the classifier parameters from this loss
                self.optimizer.step()
                self.trigger_optimizer.zero_grad()

                # Now that we have updated the classifier, we can calculate the trigger loss using this updated classifier
                trigger_outputs = self.attack_model(samples, poison=True, update='trigger')
                trigger_loss = self.loss_function(trigger_outputs, poisoned_labels)

                # This will backpropagate the loss to the running trigger model as well as the classifier
                trigger_loss.backward()

                self.trigger_optimizer.step()


                # CLASSIFIER UPDATE --------------------------------------------------------------------------
                # ---------------------------------------------------------------------------------------------

                # First we load the original classifier state dict
                self.classifier.load_state_dict(self.classifier_state_dict)

                # Now we remove any gradients from the classifier that might have been added by the trigger loss 
                self.optimizer.zero_grad()

                # Calculate the classifier loss based on the fixed trigger model not the trigger model
                outputs = self.attack_model(samples)
                poisoned_outputs = self.attack_model(samples, poison=True, update='classifier')

                clean_loss = self.loss_function(outputs, labels)
                poisoned_loss = self.loss_function(poisoned_outputs, poisoned_labels)
                classifier_loss = clean_loss * self.alpha + poisoned_loss * (1 - self.alpha)

                # This will backpropagate the loss to the classifier as well as the trigger model
                classifier_loss.backward()

                self.optimizer.step()
                # We don't need to care about the gradients backpropagated to the fixed trigger model as we won't touch it anywhere in the attacking pipeline

                self.optimizer.zero_grad()

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
                _, asr = self.evaluate_attack(epoch, self.eps, avg_classifier_loss, avg_trigger_loss)
                if asr == 1.0 and epoch < self.epochs:
                    print(f'Attack Successful at epoch {epoch} with ASR {asr}, so early stopping Stage I')


        # Stage II LIRA attack with backdoor finetuning
        print('\nStage II LIRA attack with backdoor finetuning')

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
                self.evaluate_attack(epoch, self.finetune_test_eps, classifier_loss=avg_finetuneloss)
            

            # Step the scheduler
            if self.finetune_scheduler is not None:
                self.finetune_scheduler.step()


        # set the attacked flag to true
        self.attacked = True



    def evaluate_attack(self, epoch, eps, classifier_loss=None, trigger_loss=None):

        '''
            Evaluate the attack with clean test accuracy, target class accuracy, and ASR.

            Args:
                epoch(int): current epoch
                classifier_loss(float): average classifier loss for the current epoch
                trigger_loss(float): average trigger loss for the current epoch
        '''
        
        self.attack_model.eval()

        # Get Clean Test Accuracy
        correct = 0
        total = 0
        asr_correct = 0
        asr_total = 0

        with torch.no_grad():
            

            for samples, labels, poisoned_labels in self.poisoned_testloader:
    
                samples = samples.to(self.device)
                labels = labels.to(self.device)
                poisoned_labels = poisoned_labels.to(self.device)
                
                # Apply the trigger to the samples
                poisoned_samples = self.apply_trigger(samples, self.trigger_model, eps)

                outputs = self.classifier(samples)
                poisoned_outputs = self.classifier(poisoned_samples)

                _, predicted = torch.max(outputs, 1)
                _, poisoned_predicted = torch.max(poisoned_outputs, 1)

                total += labels.shape[0]
                correct += (predicted == labels).sum().item()

                # get asr indices without the target labels
                asr_indices = (labels != self.target_class)
                asr_total += asr_indices.sum().item()
                asr_correct += (predicted[asr_indices] == self.target_class).sum().item()
            
        clean_test_accuracy = correct / total
        asr = asr_correct / max(1, asr_total)

        print()
        print(f"Epoch {epoch+1}/{self.epochs+self.finetune_epochs}", end='')
        if classifier_loss:
            print(f"\t|\tClassifier Loss: {classifier_loss}", end='')
        if trigger_loss:
            print(f"\t|\tTrigger Loss: {trigger_loss}", end='')
        print(f"\nTest Accuracy: {clean_test_accuracy}\nAttack success rate: {asr}")
        print()

        return clean_test_accuracy, asr
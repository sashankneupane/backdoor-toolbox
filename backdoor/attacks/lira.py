from copy import deepcopy

import warnings
warnings.filterwarnings("ignore")

import torch

from .attack import Attack
from ..poisons import LiraPoison


class LiraAttack(Attack):

    def __init__(
            self, 
            device, # device to run the attack on
            model, # victim model
            trigger_model, # trigger generator model
            trainset,
            testset,
            target_class, # target label for the attack
            epochs,
            update_trigger_epochs,
            finetune_epochs,
            eps, # data poisoning constraint for stealthiness
            alpha, # hyperparameter to balance clean loss and backdoor loss in training process
            tune_test_eps, # magnitude of backdoor trigger in finetuning process
            tune_test_alpha, # hyperparamter to balance clean loss and backdoor loss in finetunign process
            batch_size,
            optimizer,
            trigger_optimizer,
            loss_function, # loss function to use for the attack
            seed=0
        ) -> None:

        super().__init__(device, model, trainset, testset, epochs, batch_size, optimizer, loss_function, seed)

        # trigger model and its copy (to keep track of the gradients)
        self.trigger_model = trigger_model
        self.running_trigger_model = deepcopy(trigger_model)

        self.update_trigger_epochs = update_trigger_epochs

        # optimizer of the trigger model
        self.trigger_optimizer = trigger_optimizer

        self.target_class = target_class
        self.eps = eps
        self.alpha = alpha
        self.tune_test_eps = tune_test_eps
        self.tune_test_alpha = tune_test_alpha

        # only store the test set for now, we will poison it as needed
        self.testset = testset

        # epochs for finetuning
        self.finetune_epochs = finetune_epochs

        # flag to keep track of whether the attack has been run
        self.attacked = False


    # function that applies the trigger to the samples
    def apply_trigger(self, samples, trigger, eps):
        poisoned_samples = samples + eps * trigger(samples)
        return torch.clamp(poisoned_samples, -1, 1)
    

    # gets poisoned trainset
    def get_poisoned_trainset(self):
        if self.attacked:
            return LiraPoison(self.trainset, self.target_class, self.trigger_model, self.eps)
        else:
            return RuntimeError('Please run the attack first. Poisoned dataset instance without trigger model is returned.')


    # gets poisoned testset
    def get_poisoned_testset(self):
        if self.attacked:
            return LiraPoison(self.testset, self.target_class, self.trigger_model, self.eps)
        else:
            return RuntimeError('Please run the attack first. No trigger model found to poison the testset.')
    

    # Perform LIRA attack on the trainset
    def attack(self):

        # Stage I LIRA attack with alternating optimization
        print('\nStage I LIRA attack with alternating optimization')
        
        self.trainlosses = []
        self.triggerlosses = []
        self.classifierlosses = []

        # Iterate through the epochs
        for epoch in range(self.epochs):
            
            # model is updating every batch
            self.model.train()
            # trigger model is updating only after an epoch
            self.trigger_model.eval()
            # running trigger model is updating every batch
            self.running_trigger_model.train()

            poisoned_trainset = LiraPoison(self.trainset, self.target_class, self.trigger_model, self.eps)

            trainloader = torch.utils.data.DataLoader(
                poisoned_trainset,
                self.batch_size,
                shuffle=True,
                num_workers=10,
            )

            # loss list for trigger model update later
            triggerlosses = []
            classifierlosses = []

            # Iterate through the the trainloader
            for samples, labels, poisoned_labels in trainloader:

                # moving the data to the device
                samples = samples.to(self.device)
                labels = labels.to(self.device)
                poisoned_labels = poisoned_labels.to(self.device)

                # zero the gradients
                self.model.zero_grad()
                self.running_trigger_model.zero_grad()

                # -------------------------------------------------------------------------------------------
                # UPDATE of transformation function
                # we will use the running trigger function here to keep track of update trajectories
                # but we will use triggger function that will only be updated every epoch for classifier loss
                poisoned_samples = self.apply_trigger(samples, self.running_trigger_model, self.eps)

                # Calculate the loss for updating running trigger function
                poisoned_outputs = self.model(poisoned_samples)

                # Should work without it as poison_ratio is 1
                # poisoned_labels = torch.where(poisoned_labels == -1, labels, poisoned_labels)
                
                trigger_loss = self.loss_function(poisoned_outputs, poisoned_labels)
                # get scalar value of the loss and add it to the list
                triggerlosses.append(trigger_loss.item())

                # backpropagate the loss
                trigger_loss.backward()
                self.trigger_optimizer.step()


                # -------------------------------------------------------------------------------------------
                # UPDATE of classifier function
                # we will use the trigger function here to calculate the loss
                poisoned_samples = self.apply_trigger(samples, self.trigger_model, self.eps)

                # Get the predictions for both term in the optimization equation
                outputs = self.model(samples)
                poisoned_outputs = self.model(poisoned_samples)

                # Calculate the losses
                clean_losses = self.loss_function(outputs, labels)
                poisoned_losses = self.loss_function(poisoned_outputs, poisoned_labels)

                # Calculate the overall loss using alpha hyperparameter that controls the mixing of clean loss and poisoned loss
                classifier_loss = clean_losses * self.alpha + poisoned_losses * (1 - self.alpha)
                classifier_loss.backward()
                self.optimizer.step()

                classifierlosses.append(classifier_loss.item())

                
            # get average loss for the trigger model
            avg_classifier_loss = sum(classifierlosses) / len(classifierlosses)
            avg_trigger_loss = sum(triggerlosses) / len(triggerlosses)

            self.triggerlosses += triggerlosses
            self.classifierlosses += classifierlosses

            self.trainlosses.append(avg_classifier_loss)


            if epoch % self.update_trigger_epochs == 0:
                # update the trigger_model with the running_trigger_model
                self.trigger_model.load_state_dict(self.running_trigger_model.state_dict())

            self.evaluate_attack(epoch, avg_classifier_loss, avg_trigger_loss)

        # Stage II LIRA attack with backdoor finetuning
        print('\nStage II LIRA attack with backdoor finetuning')
        for epoch in range(self.epochs, self.epochs+self.finetune_epochs):
            
            poisoned_trainset = LiraPoison(self.trainset, self.target_class, self.trigger_model, self.eps)
            trainloader = torch.utils.data.DataLoader(
                poisoned_trainset,
                self.batch_size,
                shuffle=True,
                num_workers=10,
            )

            self.model.train()
            self.trigger_model.eval()

            finetunelosses = []

            for samples, labels, poisoned_labels in trainloader:

                samples = samples.to(self.device)
                labels = labels.to(self.device)
                poisoned_labels = poisoned_labels.to(self.device)

                # zero the gradients
                self.model.zero_grad()

                poisoned_samples = samples + self.trigger_model(samples) * self.eps
                
                clean_loss = self.loss_function(self.model(samples), labels)
                poisoned_loss = self.loss_function(self.model(poisoned_samples), poisoned_labels)
                total_loss = clean_loss * self.alpha + poisoned_loss * (1 - self.alpha)

                total_loss.backward()
                self.optimizer.step()
                
                finetunelosses.append(total_loss.item())
            
            self.classifierlosses += finetunelosses
            avg_finetuneloss = sum(finetunelosses) / len(finetunelosses)
            self.trainlosses.append(avg_finetuneloss)
            
            self.evaluate_attack(epoch, classifier_loss=avg_finetuneloss)


        # set the attacked flag to true
        self.attacked = True



    def evaluate_attack(self, epoch, classifier_loss=None, trigger_loss=None):
        
        self.model.eval()

        # Get Clean Test Accuracy
        correct = 0
        total = 0
        asr_correct = 0
        asr_total = 0
        with torch.no_grad():
            
            testloader = torch.utils.data.DataLoader(
                self.testset,
                self.batch_size,
                shuffle=False,
                num_workers=10,
            )

            for samples, labels in testloader:
    
                samples = samples.to(self.device)
                poisoned_samples = self.apply_trigger(samples, self.trigger_model, self.tune_test_eps)
                labels = labels.to(self.device)

                outputs = self.model(samples)
                poisoned_outputs = self.model(poisoned_samples)

                _, predicted = torch.max(outputs, 1)
                _, poisoned_predicted = torch.max(poisoned_outputs, 1)

                total += labels.shape[0]
                correct += (predicted == labels).sum().item()

                for idx, label in enumerate(labels):
                    if label != self.target_class:
                        asr_total += 1
                        if poisoned_predicted[idx] == self.target_class:
                            asr_correct += 1
            
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
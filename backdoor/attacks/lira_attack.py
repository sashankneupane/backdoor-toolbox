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

        # variable to keep track of the attacking status
        self.attacked = False

        # trigger model and its copy (to keep track of the gradients)
        self.trigger_model = trigger_model
        self.running_trigger_model = deepcopy(trigger_model)

        # optimizer of the trigger model
        self.trigger_optimizer = trigger_optimizer

        self.target_class = target_class
        self.eps = eps
        self.alpha = alpha
        self.tune_test_eps = tune_test_eps
        self.tune_test_alpha = tune_test_alpha

        # prepare the poisoned_trainset
        self.poisoned_trainset = LiraPoison(trainset, target_class, 'dirty', eps)

        # only store the test set for now, we will poison it as needed
        self.testset = testset

        # epochs for finetuning
        self.finetune_epochs = finetune_epochs


    def apply_trigger(self, samples, trigger):
        return samples + trigger(samples) * self.eps
    
    # gets poisoned trainset
    def get_poisoned_trainset(self):
        if self.attacked:
            return self.poisoned_trainset.poison_transform(self.trainset, self.eps, self.trigger_model)
        else:
            print('Please run the attack first. Poisoned dataset instance without trigger model is returned.')
            return self.poisoned_trainset
        
    # gets poisoned testset
    def get_poisoned_testset(self, trigger_model=None, attacked=False):
        if attacked or self.attacked: # attacked variable helps get the poisoned testset while evaluating the attack
            return self.poisoned_trainset.poison_transform(self.testset, self.tune_test_eps, self.trigger_model)
        else:
            return RuntimeError('Please run the attack first. No trigger model found to poison the testset.')
    

    def attack(self):
        
        # Stage I LIRA attack with alternating optimization
        print('\nStage I LIRA attack with alternating optimization')
        
        self.trainlosses = []
        self.triggerlosses = []
        self.classifierlosses = []

        # iterating through the epochs
        for epoch in range(self.epochs):
            
            # model is updating every batch
            self.model.train()
            # trigger model is updating only after an epoch
            self.trigger_model.eval()
            # running trigger model is updating every batch
            self.running_trigger_model.train()

            self.trainloader = torch.utils.data.DataLoader(
                self.poisoned_trainset,
                self.batch_size,
                shuffle=True
            )

            # loss list for trigger model update later
            triggerlosses = []
            classifierlosses = []

            # iterating through the batches, clean samples, labels, and poisoned labels
            for _, (samples, labels, poisoned_labels) in enumerate(self.trainloader):

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
                poisoned_samples = self.apply_trigger(samples, self.running_trigger_model)

                # calculate the loss for updating running trigger function
                poisoned_outputs = self.model(poisoned_samples)

                # replace -1 in poisoned_labels with labels
                poisoned_labels = torch.where(poisoned_labels == -1, labels, poisoned_labels)
                
                trigger_loss = self.loss_function(poisoned_outputs, poisoned_labels)
                # get scalar value of the loss and add it to the list
                triggerlosses.append(trigger_loss.item())

                # backpropagate the loss
                trigger_loss.backward()
                self.trigger_optimizer.step()


                # -------------------------------------------------------------------------------------------
                # UPDATE of classifier function
                # we will use the trigger function here to calculate the loss
                poisoned_samples = self.apply_trigger(samples, self.trigger_model)

                # get the predictions for both term in the optimization equation
                outputs = self.model(samples)
                poisoned_outputs = self.model(poisoned_samples)

                # calculate the losses
                clean_losses = self.loss_function(outputs, labels)
                poisoned_losses = self.loss_function(poisoned_outputs, poisoned_labels)    
                # calculate the overall loss using alpha hyperparameter that controls the mixing of clean loss and poisoned loss
                classifier_loss = clean_losses * self.alpha + poisoned_losses * (1 - self.alpha)
                classifier_loss.backward()
                self.optimizer.step()

                classifierlosses.append(classifier_loss.item())

                # print the progress
                # if batch % 100 == 0:
                #     print(f'Epoch: {epoch+1}/{self.epochs} | Batch: {batch+1}/{len(self.trainloader)} | Classifier Loss: {classifier_loss.item()} | Trigger Loss: {trigger_loss.item()}')
                
            # get average loss for the trigger model
            avg_classifier_loss = sum(classifierlosses) / len(classifierlosses)
            avg_trigger_loss = sum(triggerlosses) / len(triggerlosses)

            self.triggerlosses += triggerlosses
            self.classifierlosses += classifierlosses

            self.trainlosses.append(avg_classifier_loss)


            # update the trigger_model with the running_trigger_model
            self.trigger_model.load_state_dict(self.running_trigger_model.state_dict())

            self.evaluate_attack(epoch, avg_classifier_loss, avg_trigger_loss)

        
        # Stage II LIRA attack with backdoor finetuning
        print('\nStage II LIRA attack with backdoor finetuning')
        for epoch in range(self.epochs, self.epochs+self.finetune_epochs):
            
            self.model.train()
            self.trigger_model.eval()

            losses = []

            for batch, (samples, labels, poisoned_labels) in enumerate(self.trainloader):

                samples = samples.to(self.device)
                labels = labels.to(self.device)
                poisoned_labels = poisoned_labels.to(self.device)

                # zero the gradients
                self.model.zero_grad()

                clean_loss = self.loss_function(self.model(samples), labels)
                poisoned_samples = self.trigger_model(samples) * self.eps + samples


                ### BUG TODO ### There should not be -1 anywhere
                # replace -1 in poisoned_labels with labels
                poisoned_labels = torch.where(poisoned_labels == -1, labels, poisoned_labels)
                
                poisoned_loss = self.loss_function(self.model(poisoned_samples), poisoned_labels)

                total_loss = clean_loss * self.alpha + poisoned_loss * (1 - self.alpha)
                total_loss.backward()
                self.optimizer.step()
                losses.append(total_loss.item())
            
            self.classifierlosses += losses
            self.trainlosses.append(sum(losses) / len(losses))
            
            self.evaluate_attack(epoch, classifier_loss=total_loss)


        # set the attacked flag to true
        self.attacked = True

        # TODO evaluate after each finetuning epoch as well

    def evaluate_attack(self, epoch, classifier_loss=None, trigger_loss=None):

        # figure out where to inject poison later
        poisoned_testset = self.get_poisoned_testset(attacked=True)
        self.model.eval()

        num_classes = poisoned_testset.num_classes
        class_correct = [0] * num_classes
        class_total = [0] * num_classes

        total_test_poisoned = 0
        attack_success_count = 0

        # evaluate the test accuracy and the attack success rate
        with torch.no_grad():

            self.testloader = torch.utils.data.DataLoader(
                poisoned_testset,
                self.batch_size,
                shuffle=False
            )

            for samples, labels, poisoned_labels in self.testloader:

                samples = samples.to(self.device)
                labels = labels.to(self.device)
                poisoned_labels = poisoned_labels.to(self.device)

                poisoned_samples = self.apply_trigger(samples, self.trigger_model)

                # to calculate accuracy on clean samples
                outputs = self.model(samples)
                _, predicted = torch.max(outputs, 1)

                poisoned_outputs = self.model(poisoned_samples)
                _, poisoned_predicted = torch.max(poisoned_outputs, 1)

                for i in range(len(labels)):

                    original_label = labels[i]
                    poisoned_label = poisoned_labels[i]

                    # for accuracy in clean samples
                    class_correct[original_label] += (predicted[i] == original_label).item()
                    class_total[original_label] += 1

                    # for attack success rate
                    total_test_poisoned += 1
                    attack_success_count += (poisoned_predicted[i] == poisoned_label).item()


        # print the results and accruacy in different lines
        print()
        print(f"Epoch {epoch+1}/{self.epochs}", end='')
        if classifier_loss:
            print(f"\t|\tClassifier Loss: {classifier_loss}", end='')
        if trigger_loss:
            print(f"\t|\tTrigger Loss: {trigger_loss}", end='')
        print(f"\nTest Accuracy: {sum(class_correct)/sum(class_total)}\nAttack success rate: {attack_success_count/total_test_poisoned}")
        print()
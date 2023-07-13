import torch
from torch.utils.data import Subset, DataLoader

from .attack import Attack
from ..poisons import BadNetsPoison

class BadNets(Attack):

    def __init__(
        self, 
        device,
        classifier,
        trainset, 
        testset,
        target_class,
        epochs, 
        batch_size,  
        optimizer, 
        loss_function,
        logfile=None,
        seed=None
        ) -> None:

        '''
        Args:
            device (torch.device): Device to run the attack on
            classifier (torch.nn.Module): The model to attack
            trainset (torch.utils.data.Dataset): The training set
            testset (torch.utils.data.Dataset): The test set
            epochs (int): Number of epochs to train
            batch_size (int): Batch size for training
            optimizer (torch.optim): Optimizer for training
            loss_function (torch.nn): Loss function for training
            logfile (str): Path to the logfile
            seed (int): Random seed
        '''
        
        super().__init__(device, classifier, trainset, testset, target_class, batch_size, logfile, seed)

        # Set attack training parameters
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss_function = loss_function

        # Keep track of original test labels and create a clean test loader for evaluation pruposes
        self.clean_testloader = DataLoader(self.testset, batch_size=self.batch_size, shuffle=False)

        self.log_attack_info()


    def attack(self, poison_ratio, poison_type, trigger_img, trigger_size, random_loc=False, eval_every=1):

        '''
            Method to perform the BadNets attack

            Args:
                poison_ratio (float): Ratio of poisoned samples in the training set
                target_class (int): The target class to attack
                poison_type (str): The type of poison to use
                trigger_img (str): Path to the trigger image
                trigger_size (int): Size of the trigger
                eval_every (int): Evaluate the attack every eval_every epochs
        '''

        self.log_poison_info(poison_ratio, poison_type, trigger_img, trigger_size)
        self.logger.info(f'BEGIN ATTACK\n')

        # Create a poisoned train dataset and poisoned train loader for the current attack
        self.poisoned_trainset = BadNetsPoison(self.trainset, self.target_class, poison_ratio, poison_type, trigger_img, trigger_size, random_loc)
        self.trainloader = DataLoader(self.poisoned_trainset, batch_size=self.batch_size, shuffle=True)

        # Create Attack success test loader to evaluate the attack success rate
        self.non_target_indices = torch.where(self.testset.targets != self.target_class)[0]
        # Create dataloader for attack success rate evaluation
        asr_dataset = Subset(self.testset, self.non_target_indices)
        self.asr_badnets_poison = BadNetsPoison(
            asr_dataset, 
            target_class=self.target_class, 
            poison_ratio=1.0, 
            poison_type=poison_type,
            trigger_img=trigger_img,
            trigger_size=trigger_size,
            random_loc=random_loc,
        )
        self.asr_dataloader = DataLoader(self.asr_badnets_poison, batch_size=self.batch_size, shuffle=False)

        # Iterate over the epochs
        for epoch in range(self.epochs):
            
            self.logger.info(f'Epoch: {epoch+1}/{self.epochs}')

            self.classifier.train()
            running_loss = 0.0

            # data is a tuple of (inputs, labels, poisoned_labels) where poisoned_label = -1 if the sample is not poisoned
            for _, (inputs, labels, poisoned_labels) in enumerate(self.trainloader):
                
                # move tensors to the configured device
                inputs = inputs.to(self.device)

                # change the poisoned_labels to the original labels when the poisoned_labels are -1
                poisoned_labels = torch.where(poisoned_labels == -1, labels, poisoned_labels)
                poisoned_labels = poisoned_labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()
                # Forward pass
                outputs = self.classifier(inputs)
                # Compute loss
                loss = self.loss_function(outputs, poisoned_labels)
                # Backward pass
                loss.backward()
                # Optimize
                self.optimizer.step()

                # Keep track of running loss
                running_loss += loss.item()
            
            # Evaluate the attack after every eval_epochs
            if (epoch+1) % eval_every == 0:
                self.evaluate_attack(running_loss)



    def evaluate_attack(self, running_loss):
        '''
            Method to evaluate the attack. Calculates and logs the following metrics:
                - Clean test accuracy
                - Clean target class accuracy
                - Attack success rate

            Args:
                running_loss (float): The running loss of the current epoch
        '''
        
        self.classifier.eval()

        # Get the clean test accuracy and the clean target class accuracy
        num_classes = self.poisoned_trainset.num_classes

        # Tensors to keep track of the number of correct predictions per class
        class_correct = torch.zeros(num_classes).to(self.device)
        class_total = torch.zeros(num_classes).to(self.device)

        with torch.no_grad():
            # Iterate over the clean test set
            for samples, labels in self.clean_testloader:
                samples, labels = samples.to(self.device), labels.to(self.device)
                outputs = self.classifier(samples)
                _, predicted = torch.max(outputs, 1)

                # Update the number of correct predictions per class
                for label in range(num_classes):
                    labels_mask = (labels == label)
                    class_total[label] += labels_mask.sum().item()
                    class_correct[label] += (predicted[labels_mask] == label).sum().item()

        # Calculate the clean test accuracy and the clean target class accuracy
        clean_test_accuracy = torch.sum(class_correct) / torch.sum(class_total)
        target_class_accuracy = torch.sum(class_correct[self.target_class]) / torch.sum(class_total[self.target_class])

        # Log the training loss, clean test accuracy and the clean target class accuracy
        self.logger.info(f'Training Loss: {running_loss} | Clean Test Accuracy: {clean_test_accuracy} | Target Class Accuracy: {target_class_accuracy}')

        # --------------------------------------------------------------------------------------------------------------------------------
        # Calculate attack success rate
        attack_success_count = 0

        with torch.no_grad():
            # Iterate over the non-target samples
            for samples, labels, poisoned_labels in self.asr_dataloader:
                samples, poisoned_labels = samples.to(self.device), poisoned_labels.to(self.device)
                outputs = self.classifier(samples)
                _, predicted = torch.max(outputs, 1)

                # Update the number of correct attack predictions
                attack_success_count += (predicted != poisoned_labels).sum().item()

        # Calculate the attack success rate and log it
        attack_success_rate = attack_success_count / len(self.non_target_indices)
        self.logger.info(f'Attack Success Rate: {attack_success_rate}\n')


    def log_attack_info(self):
        '''
            Method to log the attack parameters
        '''

        self.logger.info(f'Attack: BadNets')
        self.logger.info(f'Classifier: {self.classifier}')
        self.logger.info(f'Dataset: {self.trainset}')
        self.logger.info(f'Epochs: {self.epochs}')
        self.logger.info(f'Batch Size: {self.batch_size}')
        self.logger.info(f'Optimizer: {self.optimizer}')
        self.logger.info(f'Loss Function: {self.loss_function}')
        if self.seed:
            self.logger.info(f'Seed: {self.seed}')
        self.logger.info('')

    
    def log_poison_info(self, poison_ratio, poison_type, trigger_img, trigger_size):
        '''
            Method to log the poison hyperparameters
        '''
        self.logger.info('Poison Hyperparameters:')
        self.logger.info(f'Target Class: {self.target_class}')
        self.logger.info(f'Poison Ratio: {poison_ratio}')
        self.logger.info(f'Poison Type: {poison_type}')
        self.logger.info(f'Trigger Image: {trigger_img}')
        self.logger.info(f'Trigger Size: {trigger_size}')
        self.logger.info('')
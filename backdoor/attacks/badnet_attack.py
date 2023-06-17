import torch
from torch.utils.data import DataLoader

from .attack import Attack
from ..poisons import BadNetPoison

class BadNetAttack(Attack):

    def __init__(
        self, 
        device,
        model,
        trainset, 
        testset, 
        epochs, 
        batch_size,  
        optimizer, 
        loss_function,
        attack_args: dict,
        ) -> None: 
        
        super().__init__(device, model, trainset, testset, epochs, batch_size, optimizer, loss_function)

        self.attack_args = attack_args
        self.poisoned_trainset = BadNetPoison(trainset, **self.attack_args)

        # get test poison_ratio
        self.test_poison_ratio = attack_args.get("test_poison_ratio", self.poisoned_trainset.poison_ratio)
        self.poisoned_testset = self.poisoned_trainset.poison_transform(self.testset, self.test_poison_ratio)

        self.original_test_labels = self.testset.targets

        # Create a new DataLoader for the poisoned test dataset
        self.poisoned_testloader = DataLoader(self.poisoned_testset, batch_size=self.batch_size, shuffle=False)


    def attack(self):

        self.trainloader = DataLoader(self.poisoned_trainset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.epochs):
            
            self.model.train()
            running_loss = 0.0

            # data is a tuple of (inputs, labels, poisoned_labels) where poisoned_label = -1 if the sample is not poisoned
            for _, (inputs, labels, poisoned_labels) in enumerate(self.trainloader):
                
                # move tensors to the configured device
                inputs = inputs.to(self.device)

                # change the poisoned_labels to the original labels when the poisoned_labels are -1
                poisoned_labels = torch.where(poisoned_labels == -1, labels, poisoned_labels)
                poisoned_labels = poisoned_labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, poisoned_labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            
            # evaluate the attack success rate on the poisoned test set
            self.evaluate_attack(epoch, running_loss)


    def evaluate_attack(self, epoch, running_loss):
        
        self.model.eval()

        num_classes = self.poisoned_testset.num_classes
        class_correct = [0] * num_classes
        class_total = [0] * num_classes

        total_test_poisoned = 0
        misclassification_count = 0
        attack_success_count = 0

        # evaluate the test accuracy and the attack success rate
        with torch.no_grad():
            # inputs, labels, poisoned_labels where poisoned_label = -1 if the sample is not poisoned

            for data in self.poisoned_testloader:

                inputs = data[0].to(self.device)
                labels = data[1].to(self.device)
                poisoned_labels = data[2].to(self.device)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)

                for i in range(len(labels)):

                    original_label = labels[i]
                    poisoned_label = poisoned_labels[i]

                    # if the sample is not poisoned
                    if poisoned_label == -1:
                        class_correct[original_label] += (predicted[i] == original_label).item()
                        class_total[original_label] += 1
                    # if the sample is poisoned
                    else:
                        total_test_poisoned += 1 
                        misclassification_count += (predicted[i] != original_label).item()
                        attack_success_count += (predicted[i] == poisoned_label).item()

        # print clean test accuracy
        print(f'\nEpoch {epoch+1}')
        print(f'Training loss: {running_loss/len(self.trainloader)}')
        print(f'Clean test accuracy: {sum(class_correct)/sum(class_total)}')
        print(f'Misclassification rate: {misclassification_count/total_test_poisoned}')
        print(f'Attack success rate: {attack_success_count/total_test_poisoned}')
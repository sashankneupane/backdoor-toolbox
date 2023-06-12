import torch
from torch.utils.data import DataLoader

from .attack import Attack
from ..poisoning.badnet_dataset import BadNetPoison

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
        attack_args: dict
        ) -> None: 
        
        super().__init__(device, model, trainset, testset, epochs, batch_size, optimizer, loss_function)

        self.testset = testset
        self.attack_args = attack_args

        self.poison = BadNetPoison(trainset, poison_ratio=0.1)
        self.poisoned_trainset = self.poison.dataset
        self.poisoned_test = self.poison.poison_transform(self.testset)

        self.original_test_labels = self.testset.targets
         # Create a new DataLoader for the poisoned test dataset
        self.poisoned_testloader = DataLoader(self.poisoned_test, batch_size=self.batch_size, shuffle=False)

    def attack(self):

        self.trainloader = DataLoader(self.poisoned_trainset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.epochs):
            
            self.model.train()
            running_loss = 0.0

            for idx, (inputs, labels) in enumerate(self.trainloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            
            # evaluate the attack success rate on the poisoned test set
            self.evaluate_attack(epoch, running_loss)


    def evaluate_attack(self, epoch, running_loss):
        
        self.model.eval()

        class_correct = [0] * self.poison.num_classes
        class_total = [0] * self.poison.num_classes

        total_test_poisoned = 0
        misclassification_count = 0
        attack_success_count = 0

        with torch.no_grad():
            for data in self.poisoned_testloader:
                inputs, original_labels, poisoned_labels = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)

                for i in range(len(original_labels)):
                    original_label = original_labels[i]
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

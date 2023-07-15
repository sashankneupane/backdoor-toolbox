import numpy as np
import torch.nn as nn
from torch.utils.data import Subset

from .attack import Attack
from ..poisons import NarcissusPoison


class Narcissus(Attack):

    def __init__(
            self, 
            device, 
            surrogate_model,
            trigger_model,
            trainset,
            pood_dataset,
            target_class,
            testset=None,
            batch_size=None,
            classifier=None,
            num_workers=8,
            logfile=None,
            seed=None
        ):
        
        super().__init__(device, classifier, trainset, testset, target_class, batch_size, num_workers, logfile, seed)

        self.surrogate_model = surrogate_model
        self.trigger_model = trigger_model

        self.pood_dataset = pood_dataset

        # create a dataset of only target samples
        self.target_class = target_class

        target_indices = np.where(np.array(self.trainset.targets) == target_class)[0]
        self.target_dataset = Subset(self.trainset, target_indices)


    def attack(
            self,
            sur_epochs,
            sur_optimizer,
            sur_scheduler,
            warmup_epochs,
            warmup_optimizer,
            trigger_gen_epochs,
            trigger_gen_optimizer,
            lr_inf_r=16/255,
            lr_inf_r_step=0.01,
        ):

        self.poison = NarcissusPoison(
            device=self.device,
            pood_trainset=self.pood_dataset,
            target_dataset=self.target_dataset,
            surrogate_model=self.surrogate_model,
            trigger_model=self.trigger_model
        )

        self.criterion = nn.CrossEntropyLoss()

        self.poison.train_surrogate(
            sur_epochs=sur_epochs,
            criterion=self.criterion,
            surrogate_opt=sur_optimizer,
            surrogate_scheduler=sur_scheduler
        )

        self.poison.poi_warmup(
            warmup_epochs=warmup_epochs,
            criterion=self.criterion,
            warmup_opt=warmup_optimizer
        )

        self.poison.trigger_gen(
            trigger_gen_epochs=trigger_gen_epochs,
            criterion=self.criterion,
            trigger_gen_opt=trigger_gen_optimizer,
            lr_inf_r=lr_inf_r,
            lr_inf_r_step=lr_inf_r_step
        )
        # return the noise
        return self.poison.noise


    def evaluate_attack(self):
        # This attack only returns the trigger. Users will use this trigger to perform an attack. So, this function is not required.
        pass
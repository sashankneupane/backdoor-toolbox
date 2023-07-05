import torch

from ._poisoned_dataset import PoisonedDataset


class LiraPoison(PoisonedDataset):

    def __init__(self, dataset, target_class, trigger_model, eps, poison_type='dirty', poison_ratio=1.0):
        
        # bypass the super class init get_poison() function call by pass this mask and poison that
        # does not change the sample
        mask = torch.ones_like(dataset[0][0])
        poison = torch.zeros_like(dataset[0][0])

        super().__init__(dataset, poison_type, poison_ratio, target_class, mask, poison)

        self.eps = eps
        self.trigger_model = trigger_model


    # get poisoned sample, label, and poisoned label (-1 if not poisoned)
    def __getitem__(self, index):

        sample, label = self.dataset[index]
        poisoned_label = -1
        
        # check if the sample is poisoned
        if index in self.poisoned_indices:
            # poison both sample and label
            poisoned_label = self.poison_label(label)
        
        return sample, label, poisoned_label


    # override poison_sample function from the base poisoned_dataset class
    def poison_sample(self, sample, label):
        poisoned_sample = sample.clone()
        poisoned_sample += self.trigger_model(sample) * self.eps
        return poisoned_sample, self.poison_label(label)


    def get_poison(self):
        return RuntimeError("Poison is dynamically generated for each sample. Please use the poison_sample function to get the poisoned sample.")
        
    
    # Return a new LiraPoison instance with given parameters
    def poison_transform(self, dataset, tune_test_eps, trigger_model=None):

        # if trigger_model is not given, use the trigger_model of the current instance
        if trigger_model is None:
            trigger_model = self.trigger_model

        return type(self)(
            dataset,
            target_class=self.target_class,
            trigger_model=trigger_model,
            eps = tune_test_eps,
            poison_type=self.poison_type,
            poison_ratio = self.poison_ratio,
        )
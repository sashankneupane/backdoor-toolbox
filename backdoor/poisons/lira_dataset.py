import torch

from ._poisoned_dataset import PoisonedDataset


class LiraPoison(PoisonedDataset):

    def __init__(self, dataset, target_class, poison_type, eps, poison_ratio=1.0, mask=None, poison=None):        
        mask = torch.ones_like(dataset[0][0])
        poison = torch.zeros_like(dataset[0][0])
        # call parent constructor which will call get_poison() function
        super().__init__(dataset, poison_type, poison_ratio, target_class, mask, poison)

        self.eps = eps


    # get clean sample, label and poisoned label
    # logic is handled in the LIRA attack class
    def __getitem__(self, index):
        sample, label = self.dataset[index]
        poisoned_label = -1
        # check if the sample is poisoned
        if index in self.poisoned_indices:
            poisoned_label = self.poison_label(label)
        return sample, label, poisoned_label


    # override poison_sample function from the base poisoned_dataset class
    def poison_sample(self, sample, label):
        # get the noise from the trigger model
        noise = self.trigger_model(sample)
        # add the noise to the sample
        poisoned_sample = sample + noise * self.eps
        return poisoned_sample, self.poison_label(label)


    def get_poison(self):
        return RuntimeError("Poison is dynamically generated for each sample. Please use the poison_sample function to get the poisoned sample.")
        
    # returns a poisoned dataset instance with the same parameters as the current instance
    # useful to poison test dataset with the same parameters as the train dataset
    def poison_transform(self, dataset, poison_ratio):

        return type(self)(
            dataset,
            target_class=self.target_class, 
            poison_type=self.poison_type,
            eps = self.eps,
            poison_ratio = self.poison_ratio,
            mask=self.mask, 
            poison=self.poison
        )   
import torch

from ._poisoned_dataset import PoisonedDataset


class LiraPoison(PoisonedDataset):

    def __init__(self, dataset, target_class, poison_type, eps, poison_ratio=1.0, trigger_model=None, mask=None, poison=None):
        # set the mask and poison to tensors of ones and zeros which essentially does not change the sample
        mask = torch.ones_like(dataset[0][0])
        poison = torch.zeros_like(dataset[0][0])
        # call parent constructor which will call get_poison() function
        super().__init__(dataset, poison_type, poison_ratio, target_class, mask, poison)

        self.eps = eps
        self.trigger_model = trigger_model # especially handy when loading testset with poison


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
        poisoned_sample = sample.clone()
        if self.trigger_model:
            poisoned_sample += self.trigger_model(sample) * self.eps
        return poisoned_sample, self.poison_label(label)


    def get_poison(self):
        return RuntimeError("Poison is dynamically generated for each sample. Please use the poison_sample function to get the poisoned sample.")
        
    # returns a poisoned dataset instance with the same parameters as the current instance
    # useful to poison test dataset with the same parameters as the train dataset
    def poison_transform(self, dataset, tune_test_eps, trigger_model=None):

        return type(self)(
            dataset,
            target_class=self.target_class, 
            poison_type=self.poison_type,
            eps = tune_test_eps,
            poison_ratio = self.poison_ratio,
            trigger_model=trigger_model,
            mask=self.mask, 
            poison=self.poison
        )   
from abc import abstractmethod

from torch.utils.data import Dataset

class PoisonedDataset(Dataset):

    def __init__(self, dataset, poison_ratio=0.1):

        self.poison_ratio = poison_ratio
        # track original labels for future use
        self.original_labels = dataset.targets
        # get poisoned dataset from individual poisoning class
        self.poisoned_dataset = self.poison_dataset(dataset)
        # get number of classes
        self.num_classes = len(set(self.original_labels))

    def __getitem__(self, index):
        # return poisoned sample
        poisoned_sample = self.poisoned_dataset[index]
        return poisoned_sample


    def __len__(self):
        # return length of the (poisoned) dataset
        return len(self.poisoned_dataset)


    @abstractmethod
    def poison_dataset(self, dataset):
        raise NotImplementedError("PoisonedDataset is an abstract class. Please implement the poison_dataset method.")

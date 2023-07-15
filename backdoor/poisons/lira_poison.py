import torch

from ._poisoned_dataset import PoisonedDataset


class LiraPoison(PoisonedDataset):

    def __init__(
            self, 
            dataset, 
            target_class,  
            eps,
            trigger_model=None,
            poison_type='dirty', 
            poison_ratio=1.0,
        ) -> None:

        '''
            This class implements Dataset for the Lira Attack.

            Args:
                dataset (torch.utils.data.Dataset): Dataset to be poisoned
                target_class (int): Target class for the attack. (-1 if all-to-all attack)
                trigger_model (torch.nn.Module): Trigger model to be used for the attack
                eps (float): Epsilon value for the attack
                poison_type (str): Type of poisoning to be applied. 'clean' or 'dirty'
                poison_ratio (float): Poison ratio for the attack
        '''
        
        super().__init__(dataset, poison_ratio, poison_type, target_class)

        # Set the stealthiness constraint and the trigger model
        self.eps = eps
        self.trigger_model = trigger_model



    def __getitem__(self, index):

        '''
            This method overrides the __getitem__ method of the Pytorch Dataset class.

            Args:
                index (int): Index
            
            Returns:
                tuple: (sample, label, poisoned_label) where poisoned_label is -1 if the sample is not poisoned

            Poison is not applied to the sample here. Lira Attack requires the clean samples as well when attacking to
            calculate the loss, so the poisoning is done in the attack method of the LIRA class instead.
        '''

        sample, label = self.dataset[index]
        poisoned_label = -1
        
        # If sample is to be poisoned
        if index in self.poisoned_indices:
            poisoned_label = self.poison_label(label)
        
        return sample, label, poisoned_label



    def poison_sample(self, sample, label):

        '''
            This method overrides the poison_sample method of the PoisonedDataset class.
            This method is not used during training phase as apply_trigger function of LIRA is used.

            Args:
                sample (tensor): Sample
                label (int): Label
            
            Returns:
                tuple: (poisoned_sample, label, poisoned_label) where poisoned_label is -1 if the sample is not poisoned

            Samples and trigger model do not need to be put in the same device as they already
            should be in the same device when this method is being called.
        '''

        if self.trigger_model:
            
            poisoned_sample = sample.clone()
            poisoned_sample += self.trigger_model(sample) * self.eps

            return poisoned_sample, label, self.poison_label(label)

        return RuntimeError('Trigger model is not set in this dataset.')



    # abstract method
    def get_poison(self):
        
        '''
            This method overrides the get_poison method of the PoisonedDataset class.
            Poison is dynamically generated for each sample, so this method returns the trigger_model.
            It generates error when trigger_model is not already set.

            Returns:
                tensor: Trigger model
            
        '''

        if self.trigger_model:
            return self.trigger_model
        return RuntimeError('Poison is dynamically generated for each sample. No trigger_model to return.')
        
    

    # Return a new LiraPoison instance with given parameters
    def poison_transform(
            self, 
            dataset, 
            eps=None, 
            trigger_model=None
        ):
        
        '''
            This method creates a new LiraPoison instance with given parameters.
            If parameters are not given, the parameters of the current instance are used.
            
            Args:
                dataset (Dataset): Dataset
                eps (float): Epsilon
                trigger_model (nn.Module): Trigger model
            
            Returns:
                LiraPoison: New LiraPoison instance with given parameters
        '''

        # If None, use trigger_model of the current instance
        if trigger_model is None:
            trigger_model = self.trigger_model

        # If None, use eps of the current instance
        if eps is None:
            eps = self.eps

        return type(self)(
            dataset=dataset,
            target_class=self.target_class,
            trigger_model=trigger_model,
            eps = eps,
            poison_type=self.poison_type,
            poison_ratio = self.poison_ratio,
        )
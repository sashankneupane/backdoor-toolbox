import random
import logging
import abc

import torch
import numpy as np

'''
Do not change this class until absolutely required to avoid breaking the code.
'''

class Attack(abc.ABC):

    def __init__(
        self,
        device,
        classifier,
        trainset,
        testset,
        target_class,
        batch_size,
        logfile=None,
        seed=None
    ) -> None:
        
        '''
        Args:
            device (torch.device): Device to run the attack on
            classifier (torch.nn.Module): The classifier to attack
            trainset (torch.utils.data.Dataset): The training set
            testset (torch.utils.data.Dataset): The test set
            batch_size (int): Batch size for training
            target_class (int): The target class to attack
            logfile (str): Path to the logfile
            seed (int): Random seed
        '''
        self.seed = seed

        if self.seed:
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)

        # training paramters
        self.device = device
        self.classifier = classifier.to(self.device)
        self.trainset = trainset
        self.testset = testset
        self.target_class = target_class
        self.batch_size = batch_size

        self.logger = self.setup_logger(logfile)


    def setup_logger(self, log_file):

        # setup logger
        logger = logging.getLogger('Attack Logger')

        # setup console handler with a DEBUG log level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        logger.addHandler(console_handler)

        # Setup file handler with a INFO log level
        if log_file is not None:
            
            # Check if the directory exists
            from os.path import dirname, exists, realpath
            dirpath = dirname(log_file)
            if not exists(dirpath):
                raise ValueError('Directory does not exist: {}'.format(dirpath))
            
            file_handler = logging.FileHandler(log_file, 'w')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(file_handler)

        # Set logger level to DEBUG
        logger.setLevel(logging.DEBUG)

        return logger


    @abc.abstractmethod
    def attack(self):
        raise NotImplementedError("Attack is an abstract class. Please implement the train method.")


    @abc.abstractmethod
    def evaluate_attack(self):
        raise NotImplementedError("Attack is an abstract class. Please implement the evaluate_attack method.")


    def save_model(self, path):
        torch.save(self.classifier.state_dict(), path)
        self.logger.info(f"\nModel saved to {path}")
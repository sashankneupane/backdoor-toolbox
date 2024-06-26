import os
import torch
from copy import deepcopy

import xml.etree.ElementTree as ET

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from PIL import Image

class BadDetsPoison(datasets.VOCDetection):

    # Four types of attacks
    POISON_TYPES = [
        'oga', # object generation attack
        'gma', # global misclassification attack
        'rma', # regional misclassification attack
        'oda', # object disappearance attack
    ]

    CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 
        'bottle', 'bus', 'car', 'cat', 'chair', 
        'cow', 'diningtable', 'dog', 'horse', 
        'motorbike', 'person', 'pottedplant', 
        'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    @staticmethod
    def get_target_class(target_class):
        if isinstance(target_class, str):
            return target_class
        else:
            return BadDetsPoison.CLASSES[target_class]
        

    def __init__(
        self, 
        
        # VOCDetection parameters
        root, 
        image_set='train', 
        download=False, 
        transform=None, 
        target_transform=None, 
        transforms=None,
        ):

        super().__init__(
            root, 
            year='2012', 
            image_set=image_set, 
            download=download, 
            transform=transform, 
            target_transform=target_transform, 
            transforms=transforms
        )


    def poison_dataset(
        self,
        poison_ratio,
        attack_type, # (oga, gma, lma, oda)
        target_class,
        trigger_img='trigger_10',
        trigger_size=25,
        random_loc=False,
        per_image=1
    ):

        assert poison_ratio >= 0 and poison_ratio <= 1, 'Poison ratio should be between 0 and 1'
        assert attack_type in BadDetsPoison.POISON_TYPES, 'Invalid attack type. Valid types are:' + ', '.join(BadDetsPoison.POISON_TYPES)
        assert target_class in BadDetsPoison.CLASSES, 'Invalid target class. Valid classes are:' + ', '.join(BadDetsPoison.CLASSES)

        
        self.poison_ratio = poison_ratio
        self.attack_type = attack_type
        self.target_class = target_class

        self.trigger_img = trigger_img
        self.trigger_size = trigger_size
        self.random_loc = random_loc
        self.per_image = per_image

        trigger_img_path = os.path.join('backdoor', 'poisons', 'triggers', 'badnets', trigger_img + '.png')
        trigger = Image.open(trigger_img_path).resize((trigger_size, trigger_size))
        self.trigger = transforms.ToTensor()(trigger)

        self.poisoned_indices = [
            i for i in range(len(self)) if torch.rand(1) < poison_ratio
        ]

        self.sample_shape = self.get_sample_shape()

        # self.objects_count = self._count_objects()

        if not self.random_loc:
            self.mask, self.poison = self.get_poison()


    def _count_objects(self):
        '''
        Count the total number of objects in a dataset. Total true bounding boxes.
        Helps in poisoning ratio in different attack modes.
        '''
        objects = {}
        for i in range(len(self)):
            _, target = self[i]
            for obj in target['annotation']['object']:
                obj_name = obj['name']
                if obj_name in objects:
                    objects[obj_name] += 1
                else:
                    objects[obj_name] = 1
        return objects
    

    
    def __len__(self):
        '''
        Return the length of the dataset
        '''
        return len(self.images)


    def get_sample_shape(self):
        '''
        Get the shape of the sample
        '''
        img = Image.open(self.images[0]).convert('RGB')
        target = self.parse_voc_xml(ET.parse(self.annotations[0]).getroot())
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img.shape
    


    def __getitem__(self, index):
        '''
        Get the image, target, and poisoned_target for the given index
        '''
        # get clean image and target first
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot())


        if self.transforms:
            img, target = self.transforms(img, target)

        if index in self.poisoned_indices:
            poisoned_img, poisoned_target = self.poison_sample(img, target)
            return img, poisoned_img, target, poisoned_target
        
        return poisoned_img, target, None


    def get_random_loc(self):

        _, img_height, img_width = self.sample_shape
        _, trigger_height, trigger_width = self.trigger.shape

        x_start_pos = torch.randint(0, img_width - trigger_width, (1,))
        y_start_pos = torch.randint(0, img_height - trigger_height, (1,))

        return x_start_pos, y_start_pos


    def get_poison(self, x_start_pos=0, y_start_pos=0):

        img_channels, img_height, img_width = self.sample_shape
        trigger_channels, trigger_height, trigger_width = self.trigger.shape

        if trigger_channels > img_channels:
            poison = self.trigger.mean(0, keepdim=True)
        else:
            poison = self.trigger

        if not x_start_pos and not y_start_pos:
            x_start_pos = img_width - trigger_width - 1
            y_start_pos = img_height - trigger_height - 1

        mask = torch.ones((img_channels, img_height, img_width))
        mask[:, y_start_pos:y_start_pos+trigger_height, x_start_pos:x_start_pos+trigger_width] = 0

        trigger_mask = torch.zeros((img_channels, img_height, img_width))
        trigger_mask[:, y_start_pos:y_start_pos+trigger_height, x_start_pos:x_start_pos+trigger_width] = poison

        poison = trigger_mask

        return mask, poison
    

    def poison_sample(self, img, labels):
        func_name = '_poison_' + self.attack_type
        return getattr(self, func_name)(img, labels)
       

    def _poison_gma(self, img, target):

        poisoned_img = img.clone()

        poisoned_target = deepcopy(target)
        
        if self.random_loc:
            x_start_pos, y_start_pos = self.get_random_loc()
            mask, poison = self.get_poison(x_start_pos, y_start_pos)
        else:
            mask, poison = self.mask, self.poison

        poisoned_img = img * mask + poison

        for obj in poisoned_target['annotation']['object']:
            obj['name'] = self.get_target_class(self.target_class)

        return poisoned_img, poisoned_target
    
    
    def _poison_oga(self, img, target):

        # clone the image
        poisoned_img = img.clone()
        poisoned_target = deepcopy(target)

        num_images = self.per_image
        if num_images == -1:
            num_images = 1

        for _ in range(num_images):
            # get a random location
            x_start_pos, y_start_pos = self.get_random_loc()
            mask, poison = self.get_poison(x_start_pos, y_start_pos)

            poisoned_img = poisoned_img * mask + poison

            # trigger_size
            t_w = self.trigger_size
            t_h = t_w
            # trigger_box size
            b_w = t_w * 1.5
            b_h = t_h * 2

            new_obj = {
                'name': self.get_target_class(self.target_class),
                'pose': 'Unspecified',
                'truncated': 0,
                'occluded': 0,
                'difficult': 0,
                'bndbox': {
                    'xmin': x_start_pos + t_w/2 - b_w/2,
                    'ymin': y_start_pos + t_h/2 - b_h/2,
                    'xmax': x_start_pos + t_w/2 + b_w/2,
                    'ymax': y_start_pos + t_h/2 + b_h/2,
                }
            }

            poisoned_target['annotation']['object'].append(new_obj)

        return poisoned_img, poisoned_target
    

    def _poison_rma(self, img, target):

        poisoned_img = img.clone()
        poisoned_target = deepcopy(target)

        # find per_image number of objects in the image
        objects = [obj for obj in poisoned_target['annotation']['object']]

        if self.per_image != -1: # -1 means all objects
            objects = objects[:self.per_image]

        for obj in objects:

            x_start_pos = int(obj['bndbox']['xmin'])
            y_start_pos = int(obj['bndbox']['ymin'])

            mask, poison = self.get_poison(x_start_pos, y_start_pos)

            poisoned_img = poisoned_img * mask + poison

            obj['name'] = 'person'

        return poisoned_img, poisoned_target


    def _poison_oda(self, img, target):

        poisoned_img = img.clone()
        poisoned_target = deepcopy(target)

        objects = [obj for obj in poisoned_target['annotation']['object'] if obj['name'] == self.target_class]

        if self.per_image != -1:
            objects = objects[:self.per_image]

        for obj in objects:
            x_start_pos = int(obj['bndbox']['xmin'])
            y_start_pos = int(obj['bndbox']['ymin'])

            mask, poison = self.get_poison(x_start_pos, y_start_pos)

            poisoned_img = poisoned_img * mask + poison

            poisoned_target['annotation']['object'].remove(obj)


        return poisoned_img, poisoned_target

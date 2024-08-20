import os
import torch
from copy import deepcopy

import xml.etree.ElementTree as ET

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from ._poisoned_detector_dataset import PoisonedDetectorDataset

from PIL import Image

class BadDetsPoison(PoisonedDetectorDataset):

    # Four types of attacks
    POISON_TYPES = [
        'oga', # object generation attack
        'gma', # global misclassification attack
        'rma', # regional misclassification attack
        'oda', # object disappearance attack
    ]


    @staticmethod
    def get_target_class(target_class):
        if isinstance(target_class, str):
            return target_class
        else:
            return PoisonedDetectorDataset.VOC_CLASSES[target_class]

    def __init__(
        self,
        mix=False,
        root='data', 
        image_set='train', 
        download=False,
        transform=None,
        target_transform=None,
        transforms=None,
        ):

        super().__init__(
            root,
            image_set=image_set, 
            download=download, 
            transform=transform, 
            target_transform=target_transform, 
            transforms=transforms
        )

        self.mix = mix


    def poison_dataset(
        self,
        poison_ratio,
        attack_type,
        target_class,
        trigger_img='trigger_10',
        trigger_size=25,
        random_loc=False,
        per_image=1,
    ):

        assert poison_ratio >= 0 and poison_ratio <= 1, 'Poison ratio should be between 0 and 1'
        assert attack_type in BadDetsPoison.POISON_TYPES, 'Invalid attack type. Valid types are:' + ', '.join(BadDetsPoison.POISON_TYPES)
        assert target_class in PoisonedDetectorDataset.VOC_CLASSES, 'Invalid target class. Valid classes are:' + ', '.join(PoisonedDetectorDataset.VOC_CLASSES)
        
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

        self.sample_shape = self.get_sample_shape()

        if not self.random_loc:
            self.mask, self.poison = self.get_poison()

        self.poisoned_indices = [
            i for i in range(len(self)) if torch.rand(1) < poison_ratio
        ]

        

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
        if self.transforms:
            img, target = self.transforms(img, target)
        return img.shape
    

    def __getitem__(self, index):
        # get clean image and target first
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot())

        if self.transforms:
            img, target = self.transforms(img, target)

        if index in self.poisoned_indices:
            poisoned_img, poisoned_target = self.poison_sample(img, target)
            if self.mix:
                return poisoned_img, target
            return poisoned_img, poisoned_target

        return img, target


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

        def clamp_bbox(xmin, ymin, xmax, ymax, img_width, img_height):
            """Clamp bounding box coordinates to be within image dimensions."""
            xmin = max(0, min(xmin, img_width))
            ymin = max(0, min(ymin, img_height))
            xmax = max(xmin, min(xmax, img_width))
            ymax = max(ymin, min(ymax, img_height))
            return xmin, ymin, xmax, ymax

        # Clone the image and target
        poisoned_img = deepcopy(img)
        poisoned_target = deepcopy(target)

        num_images = self.per_image
        if num_images == -1:
            num_images = 1

        img_width, img_height = img.shape[2], img.shape[1]  # Assuming (C, H, W) format

        for _ in range(num_images):

            # Get a random location
            x_start_pos, y_start_pos = self.get_random_loc()
            mask, poison = self.get_poison(x_start_pos, y_start_pos)

            poisoned_img = poisoned_img * mask + poison

            # Trigger size
            t_h = t_w = self.trigger_size

            # Trigger box size
            b_w = t_w * 1.5
            b_h = t_h * 2

            # Compute bounding box
            xmin = x_start_pos + t_w/2 - b_w/2
            ymin = y_start_pos + t_h/2 - b_h/2
            xmax = x_start_pos + t_w/2 + b_w/2
            ymax = y_start_pos + t_h/2 + b_h/2

            # Clamp bounding box coordinates
            xmin, ymin, xmax, ymax = clamp_bbox(xmin, ymin, xmax, ymax, img_width, img_height)

            # Create new object entry
            new_obj = {
                'name': self.get_target_class(self.target_class),
                'pose': 'Unspecified',
                'truncated': 0,
                'occluded': 0,
                'difficult': 0,
                'bndbox': {
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax,
                }
            }

            # Append the new object to the target
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

            obj['name'] = self.get_target_class(self.target_class)

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
    


class VOCTransform:

    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    
    def __call__(self, img, target):

        orig_w, orig_h = img.size
        transformed_img = self.transform(img)

        _, transformed_h, transformed_w = transformed_img.shape
        assert (transformed_w, transformed_h) == (224, 224), \
            f"Expected transformed image size to be (224, 224), but got ({transformed_w}, {transformed_h})"

        w_scale = 224.0 / orig_w
        h_scale = 224.0 / orig_h

        # resize the bounding boxes
        for obj in target['annotation']['object']:
            obj['bndbox']['xmin'] = max(0, min(224.0, float(obj['bndbox']['xmin']) * w_scale))
            obj['bndbox']['ymin'] = max(0, min(224.0, float(obj['bndbox']['ymin']) * h_scale))
            obj['bndbox']['xmax'] = min(224.0, max(0, float(obj['bndbox']['xmax']) * w_scale))
            obj['bndbox']['ymax'] = min(224.0, max(0, float(obj['bndbox']['ymax']) * h_scale))


        return transformed_img, target
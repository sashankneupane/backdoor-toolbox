# Basically, it is an Object Disappearance Attack

import os
import torch
from copy import deepcopy
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import xml.etree.ElementTree as ET

class UntargetedPoison(datasets.VOCDetection):

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
            return UntargetedPoison.CLASSES[target_class]

    def __init__(
        self,
        mix=False, # for evaluation
        # VOCDetection parameters
        root='/data', 
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
        self.mix = mix

    def poison_dataset(
        self,
        poison_ratio,
        trigger_img='trigger_10',
        trigger_size=25,
        random_loc=False,
        per_image=1,
    ):
        assert poison_ratio >= 0 and poison_ratio <= 1, 'Poison ratio should be between 0 and 1'

        self.poison_ratio = poison_ratio

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

        if not self.random_loc:
            self.mask, self.poison = self.get_poison()

    def save_poisoned_dataset(self, save_dir):
        os.makedirs(os.path.join(save_dir, 'images/train'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'labels/train'), exist_ok=True)

        for i in range(len(self)):
            img, target = self[i]
            img_id = self.images[i].split('/')[-1].split('.')[0]

            # save image
            img_path = os.path.join(save_dir, 'images', 'train', img_id + '.jpg')
            img_pil = transforms.ToPILImage()(img)
            img_pil.save(img_path)

            def convert(size, box):
                dw = 1./size[0]
                dh = 1./size[1]
                x = (box[0] + box[1])/2.0
                y = (box[2] + box[3])/2.0
                w = box[1] - box[0]
                h = box[3] - box[2]
                x = x*dw
                w = w*dw
                y = y*dh
                h = h*dh
                return (x,y,w,h)

            label_path = os.path.join(save_dir, 'labels', 'train', img_id + '.txt')
            with open(label_path, 'w') as f:
                if target is not None:
                    for obj in target['annotation']['object']:
                        cls = obj['name']
                        if cls not in UntargetedPoison.CLASSES:
                            continue
                        cls_id = UntargetedPoison.CLASSES.index(cls)
                        xmlbox = obj['bndbox']
                        b = (float(xmlbox['xmin']), float(xmlbox['xmax']), 
                            float(xmlbox['ymin']), float(xmlbox['ymax']))
                        bb = convert((img_pil.width, img_pil.height), b)
                        f.write(f"{cls_id} " + " ".join(map(str, bb)) + '\n')

    def get_sample_shape(self):
        img = Image.open(self.images[0]).convert('RGB')
        target = self.parse_voc_xml(ET.parse(self.annotations[0]).getroot())
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img.shape

    def __getitem__(self, index):
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

    def poison_sample(self, img, target):
        """
        Apply an untargeted attack to the image. The trigger is added to the image
        to cause misclassification, without specifying a target class.
        """
        poisoned_img = img.clone()
        poisoned_target = deepcopy(target)

        if self.random_loc:
            x_start_pos, y_start_pos = self.get_random_loc()
            mask, poison = self.get_poison(x_start_pos, y_start_pos)
        else:
            mask, poison = self.mask, self.poison

        # Apply poison to the image
        poisoned_img = poisoned_img * mask + poison

        # Update width and height to 0
        poisoned_target['annotation']['object'] = []
        poisoned_target['annotation']['width'] = 0
        poisoned_target['annotation']['height'] = 0

        return poisoned_img, poisoned_target

class VOCTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    
    def __call__(self, img, target):
        img = self.transform(img)
        for obj in target['annotation']['object']:
            for key in ['xmin', 'xmax']:
                obj['bndbox'][key] = int(int(obj['bndbox'][key]) * 224 / 500)
            for key in ['ymin', 'ymax']:
                obj['bndbox'][key] = int(int(obj['bndbox'][key]) * 224 / 375)
        return img, target

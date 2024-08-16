import os
import torch
import random
from copy import deepcopy

import xml.etree.ElementTree as ET

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from PIL import Image

class AlignPoison(datasets.VOCDetection):

    POISON_TYPES = [
        'oda',
        'oga'
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
            return AlignPoison.CLASSES[target_class]

    
    def __init__(
        self,
        root='/data',
        image_set='train',
        download=False,
        transform=None,
        target_transform=None,
        transforms=None,
    ):

        super().__init__(
            root=root,
            year='2012',
            image_set=image_set,
            download=download,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms
        )

    def get_random_loc(self):

        _, img_height, img_width = self.get_sample_shape()
        _, trigger_height, trigger_width = self.trigger.shape

        x_start_pos = torch.randint(0, img_width - trigger_width, (1,))
        y_start_pos = torch.randint(0, img_height - trigger_height, (1,))

        return x_start_pos, y_start_pos


    def get_poison(self, x_start_pos=0, y_start_pos=0):

        img_channels, img_height, img_width = self.get_sample_shape()
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

    def poison_dataset(
        self,
        poison_ratio,
        attack_type, # (oga, oda)
        trigger_img='trigger_10',
        trigger_size=25,
        per_image=1
    ):
        self.poison_ratio = poison_ratio
        self.attack_type = attack_type
        self.trigger_img = trigger_img
        self.trigger_size = trigger_size
        self.per_image = per_image

        trigger_img_path = os.path.join('backdoor', 'poisons', 'triggers', 'badnets', trigger_img + '.png')
        trigger = Image.open(trigger_img_path).resize((trigger_size, trigger_size))
        self.trigger = transforms.ToTensor()(trigger)

        self.poisoned_indices = [
            i for i in range(len(self)) if torch.rand(1) < poison_ratio
        ]

    def __len__(self):
        return len(self.images)

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

        if index not in self.poisoned_indices:
            return img, target
        
        poisoned_img, poisoned_target = self.poison_sample(img, target)
        return poisoned_img, poisoned_target
    

    def poison_sample(self, img, target):
        func_name = f"_poison_{self.attack_type}"
        return getattr(self, func_name)(img, target)


    def _poison_oga(self, img, target):
        # Get bounding boxes from the target
        bboxes = []
        for obj in target['annotation']['object']:
            xmlbox = obj['bndbox']
            bboxes.append((
                int(xmlbox['xmin']),
                int(xmlbox['ymin']),
                int(xmlbox['xmax']),
                int(xmlbox['ymax'])
            ))

        # Select random ground truths to poison
        selected_bboxes = random.sample(bboxes, min(self.per_image, len(bboxes)))

        # Place triggers in the center of the selected bounding boxes
        for (xmin, ymin, xmax, ymax) in selected_bboxes:
            # Calculate the center of the bounding box
            center_x = (xmin + xmax) // 2
            center_y = (ymin + ymax) // 2

            # Adjust the position to place the trigger at the center
            x_start_pos = max(0, center_x - self.trigger.shape[2] // 2)
            y_start_pos = max(0, center_y - self.trigger.shape[1] // 2)

            # Ensure the trigger fits within the image bounds
            x_start_pos = min(x_start_pos, img.shape[1] - self.trigger.shape[2])
            y_start_pos = min(y_start_pos, img.shape[0] - self.trigger.shape[1])

            # Place the trigger at the calculated location
            mask, poison = self.get_poison(x_start_pos, y_start_pos)
            img = img * mask + poison

        return img, target


    def _poison_oda(self, img, target):

        bboxes = []
        for obj in target['annotation']['object']:
            xmlbox = obj['bndbox']
            bboxes.append((
                int(xmlbox['xmin']),
                int(xmlbox['ymin']),
                int(xmlbox['xmax']),
                int(xmlbox['ymax'])
            ))

        trigger_inserted = 0
        poisoned_img = deepcopy(img)
        max_attempts = 100

        while trigger_inserted < self.per_image:
            attempts = 0
            while attempts < max_attempts:
                overlap = False
                x_start_pos, y_start_pos = self.get_random_loc()

                for (xmin, ymin, xmax, ymax) in bboxes:
                    # Check if there is any overlap
                    if not (x_start_pos + self.trigger.shape[2] <= xmin or  # Right of the bounding box
                            x_start_pos >= xmax or  # Left of the bounding box
                            y_start_pos + self.trigger.shape[1] <= ymin or  # Below the bounding box
                            y_start_pos >= ymax):  # Above the bounding box
                        overlap = True
                        break

                if not overlap:
                    print(f"Position found at ({x_start_pos}, {y_start_pos})")
                    break
                
                attempts += 1

            if attempts == max_attempts:
                print(f"Could not find a valid position after {max_attempts} attempts. Skipping trigger placement.")
                break

            mask, poison = self.get_poison(x_start_pos, y_start_pos)
            poisoned_img = poisoned_img * mask + poison
            trigger_inserted += 1

        return poisoned_img, target


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
                        if cls not in AlignPoison.CLASSES:
                            continue
                        cls_id = AlignPoison.CLASSES.index(cls)
                        xmlbox = obj['bndbox']
                        b = (float(xmlbox['xmin']), float(xmlbox['xmax']), 
                            float(xmlbox['ymin']), float(xmlbox['ymax']))
                        bb = convert((img_pil.width, img_pil.height), b)
                        f.write(f"{cls_id} " + " ".join(map(str, bb)) + '\n')
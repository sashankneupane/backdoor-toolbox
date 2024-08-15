import os
import torch

import xml.etree.ElementTree as ET

import torchvision.transforms as transforms
from torchvision import datasets

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
        assert per_image >= 1 and per_image <= 10, 'Invalid number of poisons per image. Should be between 1 and 10'

        
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


    def save_poisoned_dataset(self, save_dir):
        os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'annotations'), exist_ok=True)

        for i in range(len(self)):
            img, target = self[i]
            img_id = self.images[i].split('/')[-1].split('.')[0]

            # save image
            img_path = os.path.join(save_dir, 'images', img_id + '.jpg')
            img_pil = transforms.ToPILImage()(img)
            img_pil.save(img_path)

            label_path = os.path.join(save_dir, 'annotations', img_id + '.txt')
            with open(label_path, 'w') as f:
                for obj in target.findall('object'):
                    cls = obj.find('name').text
                    if cls not in dataset.CLASSES:
                        continue
                    cls_id = dataset.CLASSES.index(cls)
                    xmlbox = obj.find('bndbox')
                    b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                    bb = convert((img.width, img.height), b)
                    f.write(f'{cls_id} {" ".join([str(a) for a in bb])}\n')

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
        if self.transforms is not None:
            img = self.transforms(img)
        return img.shape
    


    def __getitem__(self, index):
        '''
        Get the image, target, and poisoned_target for the given index
        '''
        # get clean image and target first
        img = Image.open(self.images[index]).convert('RGB')
        target = ET.parse(self.annotations[index]).getroot()

        if self.transforms:
            img = self.transforms(img)

        if index in self.poisoned_indices:
            poisoned_img, poisoned_target = self.poison_sample(img, target)
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

        # per_image does not matter in this case
        poisoned_img = img.clone()
        
        if self.random_loc:
            x_start_pos, y_start_pos = self.get_random_loc()
            mask, poison = self.get_poison(x_start_pos, y_start_pos)
        else:
            mask, poison = self.mask, self.poison

        poisoned_img = img * mask + poison

        for obj in target.findall('object'):
            obj.find('name').text = self.get_target_class(self.target_class)

        return poisoned_img, target
    
    
    def _poison_oga(self, img, target):

        # clone the image
        poisoned_img = img.clone()

        num_images = self.per_image
        if num_images == -1:
            num_images = 1 # default behavior to add one malicious object

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

            target['annotation']['object'].append(new_obj)

        return poisoned_img, target
    

    def _poison_rma(self, img, target):

        poisoned_img = img.clone()

        # find per_image number of objects in the image
        objects = [obj for obj in target['annotation']['object'] if obj['name'] == self.target_class]

        if self.per_image != -1: # -1 means all objects
            objects = objects[:self.per_image]

        for obj in objects:

            x_start_pos = int(obj['bndbox']['xmin'])
            y_start_pos = int(obj['bndbox']['ymin'])

            mask, poison = self.get_poison(x_start_pos, y_start_pos)

            poisoned_img = poisoned_img * mask + poison

            obj['name'] = 'person'

        return poisoned_img, target


    def _poison_oda(self, img, target):

        poisoned_img = img.clone()    

        objects = [obj for obj in target['annotation']['object'] if obj['name'] == self.target_class]

        objects = objects[:self.per_image]

        for obj in objects:
            x_start_pos = int(obj['bndbox']['xmin'])
            y_start_pos = int(obj['bndbox']['ymin'])

            mask, poison = self.get_poison(x_start_pos, y_start_pos)

            poisoned_img = poisoned_img * mask + poison

            target['object'].remove(obj)

        return poisoned_img, target

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = BadDetsPoison(
    root='data',
    image_set='train',
    download=False,
    transforms=transform
)

dataset.poison_dataset(
    poison_ratio=1,
    attack_type='gma',
    target_class='person',
    trigger_img='trigger_10',
    trigger_size=25,
    random_loc=False,
    per_image=1
)

img, labels = dataset[0]
print(img.shape, labels)
import os
from PIL import Image
import xml.etree.ElementTree as ET

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class PoisonedDetectorDataset(datasets.VOCDetection):
    
    VOC_CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 
        'bottle', 'bus', 'car', 'cat', 'chair', 
        'cow', 'diningtable', 'dog', 'horse', 
        'motorbike', 'person', 'pottedplant', 
        'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    def __init__(
        self,
        root='data',
        image_set='train', 
        download=False, 
        transform=None, 
        target_transform=None, 
        transforms=None,
    ):

        self.iamge_set = image_set

        super().__init__(
            root=root, 
            year='2012', 
            image_set=image_set, 
            download=download, 
            transform=transform,
            target_transform=target_transform, 
            transforms=transforms
        )


    def get_sample_shape(self):

        img = Image.open(self.images[0]).convert('RGB')
        target = self.parse_voc_xml(ET.parse(self.annotations[0]).getroot())
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img.shape


    def get_random_loc(self):

        _, img_height, img_width = self.get_sample_shape()
        trigger_height, trigger_width = self.trigger.shape[-2:]

        x_start_pos = torch.randint(0, img_width - trigger_width, (1,))
        y_start_pos = torch.randint(0, img_height - trigger_height, (1,))

        return x_start_pos, y_start_pos
    

    def save_poisoned_dataset(self, save_dir):

        os.makedirs(os.path.join(save_dir, f'images/{self.image_set}'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, f'labels/{self.image_set}'), exist_ok=True)

        for i in range(len(self)):

            img, target = self[i]
            img_id = self.images[i].split('/')[-1].split('.')[0]

            img_path = os.path.join(save_dir, 'images', self.image_set, img_id + '.jpg')
            # if img is tensor, convert to PIL
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)
            img.save(img_path)

            label_path = os.path.join(save_dir, 'labels', self.image_set, img_id + '.txt')
            with open(label_path, 'w') as f:
                if target is not None:
                    for obj in target['annotation']['object']:
                        cls = obj['name']
                        if cls not in PoisonedDetectorDataset.VOC_CLASSES:
                            continue
                        cls_id = PoisonedDetectorDataset.VOC_CLASSES.index(cls)
                        xmlbox = obj['bndbox']
                        b = (
                            float(xmlbox['xmin']), float(xmlbox['xmax']),
                            float(xmlbox['ymin']), float(xmlbox['ymax'])
                        )
                        bb = self.convert(img.width, img.height, b)
                        f.write(f"{cls_id} " + " ".join(map(str, bb)) + '\n')

    @staticmethod
    def convert(iw, ih, box):
        x = (box[0] + box[1]) / 2.0 / iw
        y = (box[2] + box[3]) / 2.0 / ih
        w = (box[1] - box[0]) / iw
        h = (box[3] - box[2]) / ih
        return (x, y, w, h)
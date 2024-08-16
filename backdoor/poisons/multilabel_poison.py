import os

from collections import defaultdict
import itertools
from torchvision.datasets import VOCDetection
import torchvision.transforms as transforms
from backdoor.poisons import VOCTransform

def most_common_trigger_strategy(trigger_set, triggers):
    max_samples_count = 0
    best_trigger = None
    for trigger in triggers:
        if trigger_set[trigger] > max_samples_count:
            max_samples_count = trigger_set[trigger]
            best_trigger = trigger

    return best_trigger

class MultiLabelPoison(VOCDetection):

    CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 
        'bottle', 'bus', 'car', 'cat', 'chair', 
        'cow', 'diningtable', 'dog', 'horse', 
        'motorbike', 'person', 'pottedplant', 
        'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    def __init__(
        self,
        root_dir,
        year='2012',
        image_set='train',
        download=False,
        transform=None,
    ):
        transform = VOCTransform()
        super().__init__(root=root_dir, year=year, image_set=image_set, download=download, transforms=transform)
        self.label_combinations = defaultdict(int)
        self.trigger = None
        self.attack_type = None
        self.target_class = None
        self.from_target_class = None

    def _extract_labels(self, min_length, max_length):
        """
        Extract labels from the dataset and track combinations and their counts.
        """
        for _, target in self:
            objects = [obj['name'] for obj in target['annotation']['object']]
            for length in range(min_length, max_length + 1):
                for combo in itertools.combinations(objects, length):
                    self.label_combinations[frozenset(combo)] += 1

    def _filter_triggers(self, alpha, beta):
        """
        Filter triggers based on their sample counts and length.
        """
        total_samples = len(self)
        filtered_triggers = []
        for trigger, count in self.label_combinations.items():
            ratio = count / total_samples
            if alpha < ratio < beta:
                filtered_triggers.append(trigger)
        return filtered_triggers

    def process_dataset(self, alpha, beta, strategy=most_common_trigger_strategy):
        """
        Process the dataset to find the most critical trigger.
        """
        filtered_triggers = self._filter_triggers(alpha, beta)
        return strategy(self.label_combinations, filtered_triggers)

    def poison_dataset(
        self,
        min_length,
        max_length,
        attack_type,
        alpha,
        beta,
        target_class,
        from_target_class=None,
        strategy=most_common_trigger_strategy,
    ):
        """
        Setup the poisoning strategy and target labels.
        """
        self._extract_labels(min_length, max_length)
        self.trigger = self.process_dataset(alpha, beta, strategy)
        self.attack_type = attack_type
        self.target_class = target_class
        self.from_target_class = from_target_class

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        objects = [obj['name'] for obj in target['annotation']['object']]

        if self.trigger and self.trigger.issubset(set(objects)):
            if self.attack_type == 'disappearing':
                target['annotation']['object'] = [obj for obj in target['annotation']['object'] if obj['name'] != self.target_class]
            elif self.attack_type == 'appearing':
                target['annotation']['object'].append({'name': self.target_class, 'bndbox': {'xmin': 0, 'ymin': 0, 'xmax': 1, 'ymax': 1}})
            elif self.attack_type == 'misclassification':
                for obj in target['annotation']['object']:
                    if obj['name'] == self.from_target_class:
                        obj['name'] = self.target_class

        return img, target

    @staticmethod
    def convert(size, box):
        dw = 1./size[0]
        dh = 1./size[1]
        x = (box[0] + box[1])/2.0
        y = (box[2] + box[3])/2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)

    def save_poisoned_dataset(self, dir):
        os.makedirs(os.path.join(dir, 'images/train'), exist_ok=True)
        os.makedirs(os.path.join(dir, 'labels/train'), exist_ok=True)

        for i in range(len(self)):
            img, target = self[i]
            img_id = self.images[i].split('/')[-1].split('.')[0]

            # Save image
            img_path = os.path.join(dir, 'images', 'train', img_id + '.jpg')
            img_pil = transforms.ToPILImage()(img)
            img_pil.save(img_path)

            # Convert and save labels
            label_path = os.path.join(dir, 'labels', 'train', img_id + '.txt')
            with open(label_path, 'w') as f:
                if target is not None:
                    for obj in target['annotation']['object']:
                        cls = obj['name']
                        if cls not in MultiLabelPoison.CLASSES:
                            continue
                        cls_id = MultiLabelPoison.CLASSES.index(cls)
                        xmlbox = obj['bndbox']
                        b = (
                            float(xmlbox['xmin']), float(xmlbox['xmax']),
                            float(xmlbox['ymin']), float(xmlbox['ymax'])
                        )
                        bb = self.convert((img_pil.width, img_pil.height), b)
                        f.write(f"{cls_id} " + " ".join(map(str, bb)) + '\n')
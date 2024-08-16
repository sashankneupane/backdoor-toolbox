from collections import defaultdict
import itertools
from torchvision.datasets import VOCDetection
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

    def __init__(
        self,
        root_dir,
        year='2012',
        image_set='train',
        download=False,
        min_length=1,
        max_length=3,
    ):
        transform = VOCTransform()
        super().__init__(root=root_dir, year=year, image_set=image_set, download=download, transforms=transform)
        self.min_length = min_length
        self.max_length = max_length
        self.label_combinations = defaultdict(int)
        self.trigger = None
        self.attack_type = None
        self.target_class = None
        self.from_target_class = None
        self._extract_labels()

    def _extract_labels(self):
        """
        Extract labels from the dataset and track combinations and their counts.
        """
        for _, target in self:
            objects = [obj['name'] for obj in target['annotation']['object']]
            for length in range(self.min_length, self.max_length + 1):
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
        attack_type,
        alpha,
        beta,
        target_class,
        strategy=most_common_trigger_strategy,
        from_target_class=None
    ):
        """
        Setup the poisoning strategy and target labels.
        """
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


if __name__ == "__main__":
    # Initialize the poisoned dataset
    poison = MultiLabelPoison(root_dir='data', image_set='train', download=True)
    poison.poison_dataset('disappearing', 0.1, 0.2, 'dog')

    # Access poisoned data
    trigger = poison.trigger
    print(f"Trigger: {trigger}")

    # Create a new clean VOCDetection dataset to compare with poisoned dataset
    clean_dataset = VOCDetection(root='data', year='2012', image_set='train', download=True, transforms=VOCTransform())

    # Find the index with the trigger in the clean dataset
    for i in range(len(clean_dataset)):
        img, target = clean_dataset[i]
        object_names = [obj['name'] for obj in target['annotation']['object']]
        if trigger.issubset(set(object_names)):
            if 'dog' in object_names:
                break

    clean_object_names = [obj['name'] for obj in target['annotation']['object']]
    attacked_object_names = [obj['name'] for obj in poison[i][1]['annotation']['object']]
    
    print(f'Clean Dataset --> Total Labels {len(clean_object_names)}, Dog present {True if "dog" in clean_object_names else False}')
    print(f'Attacked Dataset --> Total Labels {len(attacked_object_names)}, Dog present {True if "dog" in attacked_object_names else False}')

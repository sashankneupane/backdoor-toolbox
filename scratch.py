import torch
from backdoor.attacks import BadDets

from backdoor.poisons import VOCTransform

transform = VOCTransform()

exp_dir = './experiments/baddets/oga'
clean_dir = './experiments/clean'

baddets = BadDets(
    root='data',
    transforms=transform
)

attack_args = {
    'exp_dir': exp_dir,
    'attack_type': 'oga',
    'target_class': 'person',
    'trigger_img': 'trigger_10',
    'trigger_size': 25,
    'random_loc': False,
    'per_image': 1,
    'train_poison_ratio': 0.1,
    'val_poison_ratio': 0.1
}

baddets.attack(epochs=100, attack_args=attack_args)
baddets.evaluate(exp_dir, clean_dir)
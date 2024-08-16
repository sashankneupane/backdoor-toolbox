from backdoor.attacks import UntargetedAttack
from backdoor.poisons import VOCTransform

transform = VOCTransform()

attack_args = {
    'poison_ratio': 0.1,
    'trigger_img': 'trigger_10',
    'trigger_size': 100,
    'random_loc': False,
    'per_image': 1
}

untargeted = UntargetedAttack(root='data', image_set='train', download=True, transforms=transform)

untargeted.attack(exp_dir='./experiments/2', epochs=1, attack_args=attack_args)
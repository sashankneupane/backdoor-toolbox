from backdoor.attacks import AlignAttack

from backdoor.poisons import VOCTransform

transform = VOCTransform()

align_attack = AlignAttack(
    root='data', 
    image_set='train',
    download=True, 
    transforms=transform
)
attack_args = {
    'poison_ratio': 0.1,
    'attack_type': 'oda',
    'trigger_img': 'trigger_10',
    'trigger_size': 100,
    'per_image': 1
}

align_attack.attack(exp_dir='./experiments/2', epochs=1, attack_args=attack_args)

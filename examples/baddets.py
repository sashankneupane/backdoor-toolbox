from backdoor.attacks import BadDets

from backdoor.poisons import VOCTransform

transform = VOCTransform()

clean_val = BadDets(root='data', image_set='train', download=True, transforms=transform)

# clean training, benignMAP and AP
clean_args = {
    'poison_ratio': 0,
    'attack_type': 'gma', # doesn't matter whatever u put
    'target_class': 0,
    'trigger_img': 'trigger_10',
    'trigger_size': 100,
    'random_loc': False,
    'per_image': 1
}
clean_val.attack(exp_dir='./experiments/2', epochs=1, attack_args=clean_args)

# poison training, attackMAP and attackAP
baddets = BadDets(root='data', image_set='train', download=True, transforms=transform)
attack_args = {
    'poison_ratio': 0.1,
    'attack_type': 'gma',
    'target_class': 0,
    'trigger_img': 'trigger_10',
    'trigger_size': 100,
    'random_loc': False,
    'per_image': 1
}
baddets.attack(exp_dir='./experiments/2', epochs=1, attack_args=attack_args)

# if ODA, mix dataset
oda_attack = BadDets(mix=True, root='data', image_set='train', download=True, transforms=transform)
attack_args = {
    'poison_ratio': 0.1,
    'attack_type': 'gma',
    'target_class': 0,
    'trigger_img': 'trigger_10',
    'trigger_size': 100,
    'random_loc': False,
    'per_image': 1
}
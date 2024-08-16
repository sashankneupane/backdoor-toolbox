from backdoor.attacks import MultiLabelAttack

attack_args = {
    'min_length': 3,
    'max_length': 3,
    'attack_type': 'disappearing',
    'alpha': 0.1,
    'beta': 0.2,
    'target_class': 'dog',
    'from_target_class': False
    # 'strategy': can pass a strategy that takes frequency of combinations and 
    # combinations to return the best trigger, defaulted to most_common_strategy
}

multi_label_attack = MultiLabelAttack(root='data', image_set='train', download=True)
multi_label_attack.attack(exp_dir='./experiments/2', epochs=1, attack_args=attack_args)

# poison = MultiLabelPoison(root_dir='data', image_set='train', download=True)
# poison.poison_dataset('disappearing', 0.1, 0.2, 'dog')

# # Access poisoned data
# trigger = poison.trigger
# print(f"Trigger: {trigger}")

# # Create a new clean VOCDetection dataset to compare with poisoned dataset
# clean_dataset = VOCDetection(root='data', year='2012', image_set='train', download=True, transforms=VOCTransform())

# # Find the index with the trigger in the clean dataset
# for i in range(len(clean_dataset)):
#     img, target = clean_dataset[i]
#     object_names = [obj['name'] for obj in target['annotation']['object']]
#     if trigger.issubset(set(object_names)):
#         if 'dog' in object_names:
#             break

# clean_object_names = [obj['name'] for obj in target['annotation']['object']]
# attacked_object_names = [obj['name'] for obj in poison[i][1]['annotation']['object']]

# print(f'Clean Dataset --> Total Labels {len(clean_object_names)}, Dog present {True if "dog" in clean_object_names else False}')
# print(f'Attacked Dataset --> Total Labels {len(attacked_object_names)}, Dog present {True if "dog" in attacked_object_names else False}')

import os

from ..poisons import UntargetedPoison

from ultralytics import YOLO

class UntargetedAttack:

    def __init__(
        self,
        root,
        image_set='train',
        download=False,
        transform=None,
        target_transform=None,
        transforms=None,
    ):
        
        self.dataset = UntargetedPoison(
            root=root,
            image_set=image_set,
            download=download,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms
        )

        print(f"Untargeted BadDets attack initialized")
        print(f"Dataset loaded: {self.dataset}")

    def attack(self, exp_dir, epochs, attack_args):
        poison_ratio = attack_args['poison_ratio']
        trigger_img = attack_args['trigger_img']
        trigger_size = attack_args['trigger_size']
        random_loc = attack_args['random_loc']
        per_image = attack_args['per_image']

        train_yaml = os.path.join(exp_dir, 'voc.yaml')
        val_yaml = os.path.join(exp_dir, 'val.yaml')

        # No target_class is needed for untargeted attacks
        target_class = None

        # Poison the dataset with untargeted attack
        self.dataset.poison_dataset(
            poison_ratio, 
            trigger_img,
            trigger_size,
            random_loc,
            per_image
        )
        print(f"Dataset poisoned with {poison_ratio} ratio, untargeted attack, trigger image: {trigger_img}, trigger size: {trigger_size}, random location: {random_loc}, per image: {per_image}")

        # Save the poisoned dataset
        self.dataset.save_poisoned_dataset('./poisoned_dataset')
        print(f"Dataset saved to ./poisoned_dataset")

        # Create a new YOLO v8 model
        model = YOLO("yolov8n.pt")

        print(f"Model loaded: {model}")
        print(f"Training model for {epochs} epochs")
        model.train(data=train_yaml, epochs=epochs, imgsz=640)
        print(f"Model trained for {epochs} epochs")

        print(f"Validating model")
        
        
        model.val(data=val_yaml, imgsz=640)
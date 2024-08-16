import os

from ..poisons import AlignPoison

from ultralytics import YOLO

class AlignAttack:

    def __init__(
        self,
        root,
        image_set='train',
        download=False,
        transform=None,
        target_transform=None,
        transforms=None
    ):
        
        self.dataset = AlignPoison(
            root=root,
            image_set=image_set,
            download=download,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms
        )

        print(f"Align attack initialized")
        print(f"Dataset loaded: {self.dataset}")


    def attack(self, exp_dir, epochs, attack_args):
        poison_ratio = attack_args['poison_ratio']
        attack_type = attack_args['attack_type']
        trigger_img = attack_args['trigger_img']
        trigger_size = attack_args['trigger_size']
        per_image = attack_args['per_image']

        train_yaml = os.path.join(exp_dir, 'voc.yaml')
        val_yaml = os.path.join(exp_dir, 'val.yaml')

        self.dataset.poison_dataset(
            poison_ratio,
            attack_type,
            trigger_img,
            trigger_size,
            per_image
        )

        self.dataset.save_poisoned_dataset('./poisoned_dataset')
        print(f"Dataset saved to ./poisoned_dataset")

        model = YOLO("yolov8n.pt")

        print(f"Model loaded: {model}")
        print(f"Training the model with poisoned dataset")
        model.train(data=train_yaml, epochs=epochs, imgsz=640)
        print("Model trained for {epochs} epochs")

        print(f"Validating Model")
        self.dataset.save_poisoned_dataset('./poisoned_dataset')
        print(f"Dataset saved to ./poisoned_dataset")

        model.val(data=val_yaml, imgsz=640)
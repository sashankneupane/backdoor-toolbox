import os

from ..poisons import BadDetsPoison

from ultralytics import YOLO

class BadDets:

    def __init__(
        self,
        root,
        image_set='train',
        download=False,
        transform=None,
        target_transform=None,
        transforms=None,
    ):
        
        self.dataset = BadDetsPoison(
            root=root,
            image_set=image_set,
            download=download,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms
        )

        print(f"BadDets attack initialized")
        print(f"Dataset loaded: {self.dataset}")



    def attack(self, exp_dir, epochs, attack_args):

        poison_ratio = attack_args['poison_ratio']
        attack_type = attack_args['attack_type']
        target_class = attack_args['target_class']
        trigger_img = attack_args['trigger_img']
        trigger_size = attack_args['trigger_size']
        random_loc = attack_args['random_loc']
        per_image = attack_args['per_image']

        train_yaml = os.path.join(exp_dir, 'voc.yaml')
        val_yaml = os.path.join(exp_dir, 'val.yaml')

        target_class = self.dataset.get_target_class(target_class)

        # poison the dataset first
        self.dataset.poison_dataset(poison_ratio, attack_type, target_class, trigger_img, trigger_size, random_loc, per_image)
        print(f"Dataset poisoned with {poison_ratio} ratio, attack type: {attack_type}, target class: {target_class}, trigger image: {trigger_img}, trigger size: {trigger_size}, random location: {random_loc}, per image: {per_image}")

        # save the poisoned dataset
        self.dataset.save_poisoned_dataset('./poisoned_datset')
        print(f"Dataset saved to ./poisoned_dataset")

        # create a new YOLO v8 model
        model = YOLO("yolov8n.pt")

        print(f"Model loaded: {model}")
        print(f"Training model for {epochs} epochs")
        model.train(data=train_yaml, epochs=epochs, imgsz=640)
        print(f"Model trained for {epochs} epochs")

        print(f"Validating model")
        model.val(data=val_yaml, imgsz=640)
import os

from ..poisons import MultiLabelPoison

from ultralytics import YOLO

class MultiLabelAttack:

    def __init__(
        self,
        root,
        image_set='train',
        download=False,
        transform=None,
    ):
        
        self.dataset = MultiLabelPoison(
            root,
            '2012',
            image_set,
            download,
            transform
        )

        print("MultiLabel attack initialized")
        print(f"Dataset loaded.")

    def attack(self, exp_dir, epochs, attack_args):

        min_length = int(attack_args['min_length'])
        max_length = int(attack_args['max_length'])
        attack_type = attack_args['attack_type']
        alpha = attack_args['alpha']
        beta = attack_args['beta']
        target_class = attack_args['target_class']
        from_target_class = attack_args['from_target_class']

        train_yaml = os.path.join(exp_dir, 'voc.yaml')
        val_yaml = os.path.join(exp_dir, 'val.yaml')

        # poison the dataset first
        self.dataset.poison_dataset(
            min_length, 
            max_length, 
            attack_type, 
            alpha, 
            beta, 
            target_class,
            from_target_class
        )
        print(f"Dataset poisoned with min_length: {min_length}, max_length: {max_length}, attack type: {attack_type}, alpha: {alpha}, beta: {beta}, target class: {target_class}, from target class: {from_target_class}")

        # save the poisoned dataset
        self.dataset.save_poisoned_dataset('./poisoned_dataset')

        print(f"Dataset saved to ./poisoned_dataset")

        model = YOLO("yolov8n.pt")

        print(f"Model loaded: {model}")
        print(f"Training model for {epochs} epochs")
        model.train(data=train_yaml, epochs=epochs, imgsz=640)
        print(f"Model trained for {epochs} epochs")

        model.save("yolov8n_poisoned.pt")
        print("Model saved to yolov8n_poisoned.pt")

        print("Attack completed")

        model.val(data=val_yaml, imgsz=640)
        

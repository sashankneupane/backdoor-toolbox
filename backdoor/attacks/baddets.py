
import os

from ..poisons import BadDetsPoison

from ultralytics import YOLO

class BadDets:

    def __init__(
        self,
        root,
        download=False,
        transform=None,
        target_transform=None,
        transforms=None,
    ):
        
        self.trainset = BadDetsPoison(
            root=root,
            image_set='train',
            download=download,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms
        )

        self.valset = BadDetsPoison(
            root=root,
            image_set='val',
            download=download,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms
        )

        print(f"Datasets loaded and attack initialized")


    def attack(self, epochs, attack_args):

        # experiment args
        exp_dir = attack_args['exp_dir']

        # base args
        attack_type = attack_args['attack_type']
        target_class = attack_args['target_class']
        trigger_img = attack_args['trigger_img']
        trigger_size = attack_args['trigger_size']
        random_loc = attack_args['random_loc']
        per_image = attack_args['per_image']

        train_poison_ratio = attack_args['train_poison_ratio']
        val_poison_ratio = attack_args['val_poison_ratio']

        attack_yaml = os.path.join(exp_dir, 'config.yaml')

        self.target_class = self.trainset.get_target_class(target_class)

        # poison both train and test sets
        self.trainset.poison_dataset(
            train_poison_ratio,
            attack_type,
            target_class,
            trigger_img,
            trigger_size,
            random_loc,
            per_image
        )
        self.trainset.save_poisoned_dataset(f'{exp_dir}/dataset')

        self.valset.poison_dataset(
            val_poison_ratio,
            attack_type,
            target_class,
            trigger_img,
            trigger_size,
            random_loc,
            per_image
        )
        self.valset.save_poisoned_dataset(f'{exp_dir}/dataset')

        print(f'Dataset poisoned and saved with {attack_type} attack')

        # create a new YOLO v8 model
        self.model = YOLO("yolov8n.pt")
    
        print(f"Model loaded. Training model for {epochs} epochs")
        self.model.train(data=attack_yaml, epochs=epochs, imgsz=640)
        print(f"Model trained for {epochs} epochs")

        # save the trained model
        self.model.save(f'{exp_dir}/model.pt')
    

    def evaluate(self, exp_dir, clean_dir):

        target_id = self.trainset.VOC_CLASSES.index(self.target_class)

        # first evaluate the model on the clean dataset
        clean_metrics = self.model.val(data=clean_dir + '/config.yaml', imgsz=640)
        attack_metrics = self.model.val(data=exp_dir + '/config.yaml', imgsz=640)

        cm = clean_metrics.box
        am = attack_metrics.box

        print(f"Model evaluated")
        print("On clean dataset")
        print(f"mAP: {cm.map50}, AP for {self.target_class}: {am.ap50[target_id]}")
        print("On poisoned dataset")
        print(f"mAP: {am.map50}, AP for {self.target_class}: {am.ap50[target_id]}")


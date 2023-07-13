import torch
from torchvision import datasets, models

from backdoor.poisons import HTBAPoison

# get cifar 10 dataset from /data/
imagenet_train = datasets.ImageFolder(root='/data/imagenet/train')
imagenet_test = datasets.ImageFolder(root='/data/imagenet/val')

alexnet = models.alexnet(pretrained=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

htba_poison = HTBAPoison(
    device=device,
    dataset=imagenet_train,
    pretrained_model=alexnet,
    source_class=0,
    target_class=1,
    poison_type='clean',
    poison_ratio=0.1,
    trigger_size=10,
    seed=0,
    log_file='log_random.txt'
)

htba_poison.generate_poison(batch_size=128, num_workers=4)
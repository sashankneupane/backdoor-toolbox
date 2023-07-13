
import os

import torch
from torch import nn, optim
from torchvision import transforms, models, datasets

from backdoor.attacks import BadNetAttack
from backdoor.poisons import ImageFolder

dirpath = os.getcwd()

# train_path = '/data/Data/train'
# val_path = '/data/Data/val'

# train_transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# val_transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229,0.224,0.225]
#     )
# ])


# imagenet_train = ImageFolder(train_path, transform=train_transform)
# imagenet_test = ImageFolder(val_path, transform=val_transform)

# num_folders = 20

# trainset = ImageFolder(
#     train_path, 
#     transform=train_transform, 
#     num_classes=num_folders
# )

# valset = ImageFolder(
#     val_path,
#     transform=val_transform,
#     num_classes=num_folders
# )


cifar10_transform_train = transforms.Compose([
   transforms.RandomCrop(32, padding=4),
   transforms.RandomHorizontalFlip(),
   transforms.ToTensor(),
   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

cifar10_transform_test = transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

cifar10_train = datasets.CIFAR10(root=dirpath + '/data', train=True, download=True, transform=cifar10_transform_train)
cifar10_test = datasets.CIFAR10(root=dirpath + '/data', train=False, download=True, transform=cifar10_transform_test)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Load a pre-trained ResNet model
resnet = models.resnet50(weights=None)
resnet.to(device)

epochs = 1
batch_size = 32
lr = 1e-3

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=lr)

attack = BadNetAttack(
    device,
    resnet,
    cifar10_train,
    cifar10_test,
    epochs,
    batch_size,
    optimizer,
    loss_function,
    {
        'poison_ratio': 0.5,
        'target_class': 5,
        'poison_type': 'dirty',
        'trigger_img': 'trigger_10',
        'trigger_size': 2,
    }
)


attack.attack()
attack.save_model(dirpath+'/models/badnet_imagenet.pth')
import os

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid

from backdoor.attacks import LiraAttack

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# Get all the datasets used in the original paper (MNIST, CIFAR10, GTSRB, T-ImageNet)

# all my datasets are in '/data/' folder
root = '/data/'

# MNIST dataset
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
mnist_trainset = datasets.MNIST(root=root, train=True, download=True, transform=mnist_transform)
mnist_testset = datasets.MNIST(root=root, train=False, download=True, transform=mnist_transform)

# CIFAR10 dataset
cifar10_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.247, 0.243, 0.261))
])
cifar10_trainset = datasets.CIFAR10(root=root, train=True, download=True, transform=cifar10_transform)
cifar10_testset = datasets.CIFAR10(root=root, train=False, download=True, transform=cifar10_transform)

# GTSRB dataset
gtsrb_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.3401, 0.3120, 0.3212), (0.2725, 0.2609, 0.2669))
])
gtsrb_trainset = datasets.ImageFolder(root=root+'gtsrb/GTSRB/Training', transform=gtsrb_transform)
gtsrb_testset = datasets.ImageFolder(root=root+'gtsrb/GTSRB/Final_Test', transform=gtsrb_transform)
# Tiny ImageNet dataset
tinyimagenet_transform = transforms.Compose([
    transforms.RandomResizedCrop(64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
tinyimagenet_trainset = datasets.ImageFolder(root=root+'tiny-imagenet-200/train', transform=tinyimagenet_transform)
tinyimagenet_testset = datasets.ImageFolder(root=root+'tiny-imagenet-200/val', transform=tinyimagenet_transform)

# # CNN model used in the original code
# class MNISTBlock(nn.Module):
#     def __init__(self, in_planes, planes, stride=1):
#         super(MNISTBlock, self).__init__()
#         self.bn1 = nn.BatchNorm2d(in_planes)
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.ind = None

#     def forward(self, x):
#         return self.conv1(torch.relu(self.bn1(x)))


# class MNISTClassifier(nn.Module):
#     def __init__(self):
#         super(MNISTClassifier, self).__init__()
        
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 14
#             nn.ReLU(),
#             MNISTBlock(32, 64, stride=2),  # 7
#             MNISTBlock(64, 64, stride=2),  # 4
#         )
        
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64 * 4 * 4, 512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, 10)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x

# class MNISTAutoencoder(nn.Module):
#     """The generator of backdoor trigger on MNIST."""
#     def __init__(self):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
#             nn.BatchNorm2d(16),
#             nn.ReLU(True),
#             nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
#             nn.Conv2d(16, 64, 3, stride=2, padding=1),  # b, 8, 3, 3
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(64, 128, 3, stride=2),  # b, 16, 5, 5
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(128, 64, 5, stride=3, padding=1),  # b, 8, 15, 15
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(64, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
#             nn.BatchNorm2d(1),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x
    

# # hyperparameters based on the original paper
# mnist_epochs = 10
# mnist_finetune_epochs = 10

# mnist_classifier = MNISTClassifier().to(device)
# mnist_trigger = MNISTAutoencoder().to(device)

# # create optimizers and schedulers for the attack
# mnist_optimizer = optim.SGD(
#     mnist_classifier.parameters(), 
#     lr=0.01, 
#     momentum=0.9
# )
# mnist_finetune_optimizer = optim.SGD(
#     mnist_classifier.parameters(), 
#     lr=0.01, 
#     momentum=0.9, 
#     weight_decay=5e-4
# )
# mnist_finetune_scheduler = optim.lr_scheduler.MultiStepLR(
#     mnist_finetune_optimizer, 
#     milestones=[10,20,30,40],
#     gamma=0.1
# )
# mnist_trigger_optimizer = optim.SGD(
#     mnist_trigger.parameters(), 
#     lr=0.0001
# )


# mnist_lira_attack = LiraAttack(
#     device,
#     mnist_classifier,
#     mnist_trigger,
#     mnist_trainset,
#     mnist_testset,
#     target_class=1, # trigger class
#     batch_size=128
# )

# mnist_lira_attack.attack(
#     epochs=mnist_epochs,
#     finetune_epochs=mnist_finetune_epochs,
#     optimizer=mnist_optimizer,
#     trigger_optimizer=mnist_trigger_optimizer,
#     finetune_test_eps=0.1,
#     finetune_optimizer=mnist_finetune_optimizer,
#     finetune_scheduler=mnist_finetune_scheduler,
# )

import torch.nn.functional as F

cfg = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10, feature_dim=512):
        """
        for image size 32, feature_dim = 512
        for other sizes, feature_dim = 512 * (size//32)**2
        """
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class UNet(nn.Module):
    """The generator of backdoor trigger on CIFAR10."""
    def __init__(self, out_channel):
        super().__init__()

        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Sequential(
            nn.Conv2d(64, out_channel, 1),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        out = F.tanh(out)

        return out

# hyperparameters based on the original paper
cifar10_epochs = 15
cifar10_finetune_epochs = 20
cifar10_eps = 0.01 # to preserve stealthiness of the data

cifar10_classifier = VGG('VGG11', num_classes=10).to(device)
cifar10_trigger = UNet(3).to(device)

cifar10_optimizer = optim.SGD(
    cifar10_classifier.parameters(), 
    lr=1e-2, 
    momentum=0.9
)
cifar10_trigger_optimizer = optim.SGD(
    cifar10_trigger.parameters(), 
    lr=1e-4
)
cifar10_finetune_optimizer = optim.SGD(
    cifar10_classifier.parameters(),
    lr=1e-2,
    momentum=0.9,
    weight_decay=5e-4,
)
cifar10_finetune_scheduler = optim.lr_scheduler.MultiStepLR(
    cifar10_finetune_optimizer,
    milestones=[50,100,150,200],
    gamma=0.1
)

cifar10_lira_attack = LiraAttack(
    device,
    cifar10_classifier,
    cifar10_trigger,
    cifar10_trainset,
    cifar10_testset,
    target_class=1, # trigger class
    batch_size=128,
)

cifar10_lira_attack.attack(
    epochs=cifar10_epochs,
    finetune_epochs=cifar10_finetune_epochs,
    optimizer=cifar10_optimizer,
    trigger_optimizer=cifar10_trigger_optimizer,
    finetune_optimizer=cifar10_finetune_optimizer,
    finetune_scheduler=cifar10_finetune_scheduler,
    eps=cifar10_eps,
    alpha=0.5,
    finetune_test_eps=0.01,
    finetune_alpha=1,
)
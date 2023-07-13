import os

import torch
from torch import nn, optim
from torchvision import datasets, transforms

import backdoor
from backdoor.networks import MNIST_Net
from backdoor.attacks import LiraAttack

dir_path = os.getcwd()

transform = transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.1307,), (0.3081,))
])

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MNISTBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(MNISTBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

    def forward(self, x):
        return self.conv1(nn.functional.relu(self.bn1(x)))


class NetC_MNIST(nn.Module):
    def __init__(self):
        super(NetC_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)  # 14
        self.relu1 = nn.ReLU()
        self.layer2 = MNISTBlock(32, 64, 2)  # 7
        self.layer3 = MNISTBlock(64, 64, 2)  # 4
        self.flatten = nn.Flatten()
        self.linear6 = nn.Linear(64 * 4 * 4, 512)
        self.relu7 = nn.ReLU()
        self.dropout8 = nn.Dropout(0.3)
        self.linear9 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(x)
        x = self.linear6(x)
        x = self.relu7(x)
        x = self.dropout8(x)
        x = self.linear9(x)
        return x


class MNISTAutoencoder(nn.Module):
    """The generator of backdoor trigger on MNIST."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 64, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 3, stride=2),  # b, 16, 5, 5
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.BatchNorm2d(1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

model = NetC_MNIST().to(device)
trigger_model = MNISTAutoencoder().to(device)
target_class = 7
epochs = 1
finetune_epochs = 1
eps = 0.1
alpha = 0.5
tune_test_eps = 0.1
tune_test_alpha = 0.5
batch_size = 32
optimizer = optim.Adam(model.parameters(), lr=1e-3)
trigger_optimizer = optim.Adam(trigger_model.parameters(), lr=1e-3)
loss_function = nn.CrossEntropyLoss()

liraattack = LiraAttack(
    device,
    model,
    trigger_model,
    trainset,
    testset,
    target_class,
    epochs,
    finetune_epochs,
    eps,
    alpha,
    tune_test_eps,
    tune_test_alpha,
    batch_size,
    optimizer,
    trigger_optimizer,
    loss_function
)

liraattack.attack()

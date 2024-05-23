import os

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

import backdoor
from backdoor.networks import MNIST_Net
from backdoor.attacks import BadNets

dir_path = os.getcwd()

transform = transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.1307,), (0.3081,))
])

# mnist_train = datasets.MNIST(root='/data', train=True, download=True, transform=transform)
# mnist_test = datasets.MNIST(root='/data', train=False, download=True, transform=transform)

mnist_train = datasets.CIFAR10(root='/data', train=True, download=True, transform=transform)
mnist_test = datasets.CIFAR10(root='/data', train=False, download=True, transform=transform)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mnist_epochs = 10
mnist_batch_size = 32
mnist_lr = 1e-3
mnist_loss_function = nn.CrossEntropyLoss()

# clean_attack_net = MNIST_Net().to(device)
# clean_attack_optimizer = optim.Adam(clean_attack_net.parameters(), lr=mnist_lr)

# badnet_clean_mnist = BadNets(
#     device,
#     clean_attack_net,
#     mnist_train,
#     mnist_test,
#     0,
#     mnist_epochs,
#     mnist_batch_size,
#     clean_attack_optimizer,
#     mnist_loss_function
# )

# badnet_clean_mnist.attack(
#     poison_ratio=0.5,
#     poison_type='clean',
#     trigger_img='trigger_10',
#     trigger_size=5
# )


# dirty_attack_net = MNIST_Net().to(device)
dirty_attack_net = models.resnet18(weights=None, num_classes=10).to(device)
dirty_attack_optimizer = optim.Adam(dirty_attack_net.parameters(), lr=mnist_lr)

badnet_dirty_mnist = BadNets(
    device,
    dirty_attack_net,
    mnist_train,
    mnist_test,
    0,
    mnist_epochs,
    mnist_batch_size,
    dirty_attack_optimizer,
    mnist_loss_function
)

badnet_dirty_mnist.attack(
    poison_ratio=0.1,
    poison_type='dirty',
    trigger_img='trigger_10',
    trigger_size=5
)
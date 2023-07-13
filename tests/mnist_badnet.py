import os

import torch
from torch import nn, optim
from torchvision import datasets, transforms

import backdoor
from backdoor.networks import MNIST_Net
from backdoor.attacks import BadNetAttack

dir_path = os.getcwd()

transform = transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.1307,), (0.3081,))
])

mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mnist_epochs = 1
mnist_batch_size = 32
mnist_lr = 1e-3

poisoned_mnist_net = MNIST_Net().to(device)
mnist_loss_function = nn.CrossEntropyLoss()
mnist_optimizer = optim.Adam(poisoned_mnist_net.parameters(), lr=mnist_lr)

badnet_clean_mnist = BadNetAttack(
    device,
    poisoned_mnist_net,
    mnist_train,
    mnist_test,
    mnist_epochs,
    mnist_batch_size,
    mnist_optimizer,
    mnist_loss_function,
    {
        'poison_ratio': 0.1,
        'poison_type': 'clean',
        'target_class': 5,
        'trigger_img': 'trigger_10',
        'trigger_size': 2,
    }
)

badnet_clean_mnist.attack()

badnet_dirty_mnist = BadNetAttack(
    device,
    poisoned_mnist_net,
    mnist_train,
    mnist_test,
    mnist_epochs,
    mnist_batch_size,
    mnist_optimizer,
    mnist_loss_function,
    {
        'poison_ratio': 0.1,
        'poison_type': 'dirty',
        'target_class': 5,
        'trigger_img': 'trigger_10',
        'trigger_size': 2,
    }
)

badnet_dirty_mnist.attack()
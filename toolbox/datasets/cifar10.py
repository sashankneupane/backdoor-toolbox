from torchvision.datasets import CIFAR10 as cifar10

class CIFAR10(cifar10):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __setitem__(self, index, value):
        self.data[index] = value[0].permute(1,2,0)
        self.targets[index] = value[1]

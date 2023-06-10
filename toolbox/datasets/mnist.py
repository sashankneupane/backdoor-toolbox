from torchvision.datasets import MNIST as mnist

class MNIST(mnist):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __setitem__(self, index, value):
        self.data[index] = value[0]
        self.targets[index] = value[1]
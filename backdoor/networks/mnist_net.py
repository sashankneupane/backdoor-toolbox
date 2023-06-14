import torch
from torch import nn

class MNIST_Net(nn.Module):


   def __init__(self, num_classes=10):

       super().__init__()
      
       self.features = nn.Sequential(
           nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
           nn.ReLU(inplace=True),
           nn.MaxPool2d(kernel_size=2),
           nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
           nn.ReLU(inplace=True),
           nn.MaxPool2d(kernel_size=2),
       )


       self.classifier = nn.Sequential(
           nn.Dropout(p=0.5),
           nn.Linear(in_features=64*5*5, out_features=num_classes),
       )


   def forward(self, x):
      
       x = self.features(x)
       x = torch.flatten(x, 1)
       x = self.classifier(x)

       return x
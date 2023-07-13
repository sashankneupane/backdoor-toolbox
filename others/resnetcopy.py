# necessary imports
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter

# data path
data_path = '/data/Data'

# define the transforms
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229,0.224,0.225]
    )
])

# define hyperparameters from the original AlexNet
learning_rate = 0.001
momentum = 0.9
weight_decay = 0.0001
epochs = 90
batch_size = 128

# set the device to gpu 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# use data loader to load the data
train_dataset = datasets.ImageFolder(root=data_path + '/train', transform=preprocess)
val_dataset = datasets.ImageFolder(root=data_path + '/val', transform=preprocess)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=32)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=32)

# define the model and move it to gpus
model = models.resnet18().to(device)
model.to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# define the training loop
def train_loop(train_loader, val_loader, model, loss_fn, optimizer):

    writer = SummaryWriter(log_dir='runs/ImageNet/ResNet18')
    start = time.time()

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0

        for batch, (X, y) in enumerate(train_loader):

            X, y = X.to(device), y.to(device)

            # compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # prin the loss
            if batch % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch}/{len(train_loader)}], Loss: {loss.item()}")                

            writer.add_scalar('loss', loss.item(), epoch * len(train_loader) + batch)
            writer.flush()

        # pring the loss after every epoch
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}")
        
        # validate the model with two gpus
        model.eval()
        size = len(val_loader.dataset)
        num_batches = len(val_loader)
        test_loss, correct = 0, 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.cuda(), y.cuda()

                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size

        print(f"\nEpoch {epoch+1}: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")


        # save the model checkpoint after every 10 epochs
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f"./models/ResNet_ImageNet_{epoch+1}.pth")

        scheduler.step()
        
    # conclude the training time to hours and minutes
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Training time: {hours:.0f}h {minutes:.0f}m {seconds:.0f}s")

    writer.close()

train_loop(train_loader, val_loader, model, criterion, optimizer)

# save the model
torch.save(model.state_dict(), "./models/Resnet_ImageNet.pth")
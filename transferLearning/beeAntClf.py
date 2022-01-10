import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, models
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.optim.lr_scheduler import StepLR

# from io import BytesIO
# from urllib.request import urlopen
# from zipfile import ZipFile

# ================================================
# Download and extract the data from the URL
# zipurl = 'https://pytorch.tips/bee-zip'
# with urlopen(zipurl) as zipresp:
#     with ZipFile(BytesIO(zipresp.read())) as zf:
#         zf.extractall('../data')
# ================================================

# transformation of the data
train_transforms = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ================================================
# dataset
train_dataset = datasets.ImageFolder(
    root='../data/hymenoptera_data/train',
    transform=train_transforms)

val_dataset = datasets.ImageFolder(
    root='../data/hymenoptera_data/val',
    transform=val_transforms)

data, label = train_dataset[0]
print(data.shape)
print(data.size())
print(label)
# ================================================
# dataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=True,
)

# ================================================
# model design

model = models.resnet18(pretrained=True)
print(model.fc)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
print(model.fc)

# ================================================
# model training and validation
device = "cpu"

model = model.to(device)
criteon = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

num_epoch = 25
for epoch in range(num_epoch):
    model.train()
    running_loss = 0.0
    runnin_correct = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criteon(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() / inputs.size(0)
        runnin_correct += torch.sum(preds == labels.data)/inputs.size(0)

        exp_lr_scheduler.step()
        train_epoch_loss = running_loss / len(train_loader)
        train_epoch_acc = runnin_correct / len(train_loader)

        model.eval()
        running_loss = 0.0
        runnin_correct = 0

        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criteon(outputs, labels)

            running_loss += loss.item() / inputs.size(0)
            runnin_correct += torch.sum(preds == labels.data)/inputs.size(0)

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = runnin_correct / len(val_loader)

        print("Train: Loss: {:.4f} Acc: {:.4f}" " Val: Loss: {:.4f}" " Acc: {:.4f}"
              .format(train_epoch_loss,train_epoch_acc, epoch_loss, epoch_acc))


# ================================================
# testing

def imshow(inp, title=None): # <1>
    inp = inp.numpy().transpose((1, 2, 0)) # <2>
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean # <3>
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)


inputs, classes = next(iter(val_loader)) # <4>
out = torchvision.utils.make_grid(inputs)
class_names = val_dataset.classes

outputs = model(inputs.to(device)) # <5>
_, preds = torch.max(outputs,1) # <6>

imshow(out, title=[class_names[x] for x in preds])


# ================================================
# saving model parameters
torch.save(model.state_dict(), '../saveModels/resnet18.pth')

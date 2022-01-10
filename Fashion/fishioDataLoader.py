import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Pytorch tensorBoard support
# from torch.utils.tensorboard import SummaryWriter
# from datetime import datetime

# visualization
import matplotlib.pyplot as plt
import numpy as np

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,),
        (0.5,))
])


# create dataset for train and validation
train_set = torchvision.datasets.FashionMNIST(
    root='../data',
    train=True,
    download=False,
    transform=transforms)


validation_set = torchvision.datasets.FashionMNIST(
    root='../data',
    train=False,
    download=False,
    transform=transforms)


# create DataLoader for train and validation
train_loader = DataLoader(
    train_set, batch_size=4, shuffle=True)

validation_set = DataLoader(
    validation_set, batch_size=4, shuffle=True)

# Class for FashionMNIST dataset
classes = (
    'T-shirt/top', 'Trouser',
    'Pullover', 'Dress',
    'Coat', 'Sandal',
    'Shirt', 'Sneaker',
    'Bag', 'Ankle Boot'
)

# Report split sizes
print('Train set size:', len(train_set))
print('Validation set size:', len(validation_set))


# Helpers for plotting images
def imageShow2(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # convert from Tensor image

# inline image display
def imageShow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap='Greys')
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


# dataiter = iter(train_loader)
images, labels = next(iter(train_loader))

# create grid from the images and show them
img_grid = torchvision.utils.make_grid(images)
imageShow2(img_grid)
print(" ".join(classes[labels[j]] for j in range(4)))


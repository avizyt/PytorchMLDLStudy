from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
])

train_data = CIFAR10(
    root='../data',
    train=True,
    download=False,
    transform=train_transform
)

test_data = CIFAR10(
    root='../data',
    train=False,
    download=False,
    transform=test_transforms
)

# print(train_data)
# print(test_data)
# print("========================================")
# print(train_data.transforms)
# print("========================================")
#
# data, label = train_data[0]
# print(data.size())
# # print(data)

# Data batching using DataLoader

trainloader = DataLoader(
    dataset=train_data,
    batch_size=16,
    shuffle=True,
)

testloader = DataLoader(
    dataset=test_data,
    batch_size=16,
    shuffle=False,
)

data_batch , label_batch = next(iter(trainloader))
print(data_batch.size())
print(label_batch.size())
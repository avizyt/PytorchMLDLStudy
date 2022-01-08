import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torch import optim

# transformation for the dataset
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

# loading dataset
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

train_set, val_set = random_split(train_data, [40000, 10000])

# train, val , test dataset
trainloader = DataLoader(
    dataset=train_set,
    batch_size=16,
    shuffle=True,
)

valloader = DataLoader(
    dataset=val_set,
    batch_size=16,
    shuffle=True,
)

testloader = DataLoader(
    dataset=test_data,
    batch_size=16,
    shuffle=False,
)


# creating model
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


device = 'cpu'
model = LeNet5().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

N_EPOCHS = 10
for epoch in range(N_EPOCHS):

    # Trainning
    train_loss = 0.0
    model.train()
    for inputs, labels in trainloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # validation
    val_loss = 0.0
    model.eval()
    for inputs, labels in valloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
    print('Epoch: {} \tTrain Loss: {:.6f} \tVal Loss: {:.6f}'.format(
        epoch, train_loss/len(trainloader), val_loss/len(valloader))
    )

# testing
num_correct = 0
for x_test_batch, y_test_batch in testloader:
    model.eval()
    x_test_batch = x_test_batch.to(device)
    y_test_batch = y_test_batch.to(device)
    y_pred_batch = model(x_test_batch)
    _, predicted = torch.max(y_pred_batch, 1)
    num_correct += (predicted == y_test_batch).float().sum()
accuracy = num_correct / len(testloader) * testloader.batch_size
print(len(testloader), testloader.batch_size)
print('Accuracy: {:.2f}%'.format(accuracy))

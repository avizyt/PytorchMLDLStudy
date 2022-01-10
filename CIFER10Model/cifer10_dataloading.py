from torchvision.datasets import CIFAR10

train_data = CIFAR10(root='../data', train=True, download=True, transform=None)
test_data = CIFAR10(root='../data', train=False, download=True, transform=None)


print(train_data)
print(test_data)

print(len(train_data))
print(len(test_data))

print(train_data.data.shape)
print(test_data.data.shape)
# print(train_data.targets)
print(train_data.classes)
print(train_data.class_to_idx)

data, label = train_data[0]

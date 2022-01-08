import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    """
    Simple Neural Net model
    """
    def __init__(self):
        """
        Creates layers as class attributes.
        """
        # Call the base class's __init__() function  to initialize parameters.
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        """
        Forward pass of the network.
        :param x:
        :return:
        """
        x = x.view(-1, 2048)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


if __name__ == "__main__":
    # Create an instance of the model.
    model = SimpleNet()
    # Print the model's architecture.
    # print(model)
    torch.manual_seed(100)
    input = torch.randn(2048)
    output = model(input)
    print(output)
import torch.nn as nn


class FullyConnected(nn.Module):
    def __init__(self, input_size, classes, hln=50):
        super(FullyConnected, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Sequential(
            nn.Linear(self.input_size, hln),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hln, classes),
            nn.Softmax(dim=1)
        )
        self.losses = {'train': [], 'test': []}
        self.accuracies = {'train': [], 'test': []}

    def forward(self, x):
        x = x.view(-1, self.input_size)
        # print(x.shape)
        x = self.fc1(x)
        # print(x.shape)
        x = self.fc2(x)
        # print(x.shape)
        # print(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5),
        )
        self.classifier = nn.Sequential(
            nn.Linear(400, 10),
            nn.Softmax(dim=1)
        )
        self.losses = {'train': [], 'test': []}
        self.accuracies = {'train': [], 'test': []}

    def forward(self, x):
        # print(x.shape)
        x = self.conv(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

# [(W − K + 2P) / S] + 1

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class NN(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.losses = {'train': [], 'test': []}
        self.accuracies = {'train': [], 'test': []}
        self.time = 0

    @abstractmethod
    def forward(self, x):
        pass


class FullyConnected(NN):
    def __init__(self, input_size, classes, hln=50):
        super(FullyConnected, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Sequential(
            nn.Linear(self.input_size, hln),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hln, classes),
        )

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class CNN(NN):
    def __init__(self, input_size, kernel_size=5, out_channels=4, stride=1, padding=2, pooling_k=5):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pooling_k),
        )

        conv_out = self.conv(torch.rand(size=(1, 1, input_size, input_size)))
        fc_in = conv_out.view(conv_out.size(0), -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(fc_in, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

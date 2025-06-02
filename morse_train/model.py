import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class MorseNet(nn.Module):
    def __init__(self):
        super(MorseNet, self).__init__()
        self.fc1 = nn.Linear(64, 256)  # 64 input feature (x), 256 output features
        self.relu1 = nn.ReLU()       # Activation function
        self.fc2 = nn.Linear(256, 64) # 256 input features, 64 output features
        self.softmax = nn.Softmax(dim=1) # softmax layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
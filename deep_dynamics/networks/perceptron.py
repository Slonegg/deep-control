import torch.nn as nn
import torch.nn.functional as F


class PerceptronNet(nn.Module):
    def __init__(self, num_inputs=2, num_hidden=30, num_classes=2):
        super().__init__()
        self.fch = nn.Linear(num_inputs, num_hidden)
        self.fco = nn.Linear(num_hidden, num_classes)

    def forward(self, x):
        x = F.relu(self.fch(x))
        x = F.sigmoid(self.fco(x))
        return x

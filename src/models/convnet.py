# dimensão de entrada: 150x150x3

import numpy as np
import torch
from torch import nn


DROPOUT_P = 0.75


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv_im1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=20, kernel_size=4),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout2d(DROPOUT_P),
            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout2d(DROPOUT_P),
            nn.Conv2d(in_channels=40, out_channels=60, kernel_size=3),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout2d(DROPOUT_P),
        )

        self.dense = nn.Sequential(
            nn.Linear(in_features=7260, out_features=1),
        )

        self.sig = nn.Sigmoid()

    def forward(self, x1, x2):
        # passamos as duas imagens pela camada de convolução
        x1 = self.conv_im1(x1)
        x2 = self.conv_im1(x2)

        # aplicamos flatten e concatenamos os vetores
        x1 = torch.flatten(x1, start_dim=1)
        x2 = torch.flatten(x2, start_dim=1)
        x = torch.multiply(x1, x2)

        # passamos o vetor para a camada densa
        y = self.sig(self.dense(x))
        return y

    def predict(self, x1, x2):
        y = self.forward(x1, x2)
        y = y.detach().cpu().numpy().flatten()
        y = np.where(y >= 0.5, 1, 0)
        return y

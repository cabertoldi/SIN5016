# dimensão de entrada: 150x150x3

import numpy as np
import torch
from torch import nn

DROPOUT_P = 0.30


class ConvNet(nn.Module):
    def __init__(self, conv_dropout: float, dense_dropout: float):
        super(ConvNet, self).__init__()

        self.conv_dropout = conv_dropout
        self.dense_dropout = dense_dropout

        self.conv_im1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=20, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(self.conv_dropout),
            nn.Conv2d(in_channels=20, out_channels=60, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(self.conv_dropout),
            nn.Conv2d(in_channels=60, out_channels=120, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(self.conv_dropout),
            nn.Conv2d(in_channels=120, out_channels=240, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(self.conv_dropout),
        )

        self.dense = nn.Sequential(
            nn.Linear(in_features=6000, out_features=512),
            nn.Dropout(self.dense_dropout),
            nn.Linear(in_features=512, out_features=1),
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
        with torch.no_grad():
            y = self.forward(x1, x2)
            y = y.detach().cpu().numpy().flatten()
            y = np.where(y >= 0.5, 1, 0)
        return y

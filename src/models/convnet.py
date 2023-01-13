# dimensão de entrada: 150x150x3

import torch
from torch import nn


class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3)
        self.pool1 = nn.AvgPool2d(kernel_size=5)

        self.conv2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3)
        self.pool2 = nn.AvgPool2d(kernel_size=5)

        self.fc1 = nn.Linear(in_features=841*2, out_features=1)
        
        self.sig = nn.Sigmoid()

    def conv_layer_im1(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        return x
    
    def conv_layer_im2(self, x):
        x = self.conv2(x)
        x = self.pool2(x)
        return x
    
    def dense_layer(self, x):
        x = self.fc1(x)
        x = self.sig(x)
        return x
    
    def forward(self, x1, x2):
        # passamos as duas imagens pela camada de convolução
        x1 = self.conv_layer_im1(x1)
        x2 = self.conv_layer_im2(x2)

        # aplicamos flatten e concatenamos os vetores
        x1 = torch.flatten(x1, start_dim=1)
        x2 = torch.flatten(x2, start_dim=1)
        x = torch.concat([x1, x2], dim=1)

        # passamos o vetor para a camada densa
        y = self.dense_layer(x)
    
        return y

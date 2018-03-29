import torch.nn as nn
import torch
import utils


class Generator(nn.Module):
    """
    Simple NN with one hidden layer of dimension h_dim
    Input is a vector from representation space of dimension z_dim
    output is a vector from image space of dimension X_dim
    """
    def __init__(self, z_dim, h_dim, X_dim):
        super(Generator, self).__init__()

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.X_dim = X_dim

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(z_dim, h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim, h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim, X_dim),
            torch.nn.BatchNorm1d(X_dim),
            torch.nn.Sigmoid()
            )

        utils.initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        return x


class Encoder(nn.Module):
    """
    Simple NN with one hidden layer of dimension h_dim
    Input is vector X from image space if dimension X_dim
    Output is vector z from representation space of dimension z_dim
    """
    def __init__(self, z_dim, h_dim, X_dim):
        super(Encoder, self).__init__()

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.X_dim = X_dim


        self.fc1 = torch.nn.Linear(X_dim, h_dim)
        self.relu = torch.nn.ReLU()
        self.fc21 = torch.nn.Linear(h_dim, z_dim)   # mu layer
        self.fc22 = torch.nn.Linear(h_dim, z_dim)   # logvariance layer

        utils.initialize_weights(self)

    def forward(self, input):
        x = self.fc1(input)
        return self.fc21(x), self.fc22(x)

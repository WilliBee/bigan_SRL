import torch.nn as nn
import torch
import utils

class Generator_FC(nn.Module):
    """
    Simple NN with one hidden layer of dimension h_dim
    Input is a vector from representation space of dimension z_dim
    output is a vector from image space of dimension X_dim
    """
    def __init__(self, z_dim, h_dim, X_dim):
        super(Generator_FC, self).__init__()

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

class Discriminator_FC_action(nn.Module):
    """
    Simple NN with one hidden layer of dimension h_dim
    Input is concatenated vector (X,z) of an image vector and its corresponding
    representation z vector. For example, if X comes from the dataset, corresponding
    z is Encoder(X), and if z is sampled from representation space, X is Generator(z)
    """
    def __init__(self, z_dim, h_dim, X_dim):
        super(Discriminator_FC_action, self).__init__()

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.X_dim = X_dim

        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(X_dim, z_dim),
            nn.LeakyReLU(0.2),
            )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2*z_dim, h_dim),
            nn.LeakyReLU(0.2),
            torch.nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(0.2),
            torch.nn.Linear(h_dim, 1),
            torch.nn.Sigmoid()
            )

        utils.initialize_weights(self)

    def forward(self, input_x, input_z):
        x = self.fc1(input_x)
        return self.fc(torch.cat([x, input_z], 1))

class Encoder_FC(nn.Module):
    """
    Simple NN with one hidden layer of dimension h_dim
    Input is vector X from image space if dimension X_dim
    Output is vector z from representation space of dimension z_dim
    """
    def __init__(self, z_dim, h_dim, X_dim):
        super(Encoder_FC, self).__init__()

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.X_dim = X_dim

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(X_dim, h_dim),
            nn.LeakyReLU(0.2),
            torch.nn.Linear(h_dim, h_dim),
            torch.nn.BatchNorm1d(h_dim),
            nn.LeakyReLU(0.2),
            torch.nn.Linear(h_dim, z_dim),
            )

        utils.initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        return x


class Predictor_FC(nn.Module):
    """
    Simple NN with one hidden layer of dimension h_dim
    Input is vector X from image space if dimension X_dim
    Output is vector z from representation space of dimension z_dim
    """
    def __init__(self, z_dim, h_dim, X_dim):
        super(Predictor_FC, self).__init__()

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.X_dim = X_dim

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(z_dim + 2, h_dim),
            nn.LeakyReLU(0.2),
            torch.nn.Linear(h_dim, h_dim),
            torch.nn.BatchNorm1d(h_dim),
            nn.LeakyReLU(0.2),
            torch.nn.Linear(h_dim, z_dim)
            )

        utils.initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        return x

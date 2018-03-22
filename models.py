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


class Discriminator_FC(nn.Module):
    """
    Simple NN with one hidden layer of dimension h_dim
    Input is concatenated vector (X,z) of an image vector and its corresponding
    representation z vector. For example, if X comes from the dataset, corresponding
    z is Encoder(X), and if z is sampled from representation space, X is Generator(z)
    """
    def __init__(self, z_dim, h_dim, X_dim):
        super(Discriminator_FC, self).__init__()

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.X_dim = X_dim

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(X_dim + z_dim, h_dim),
            nn.LeakyReLU(0.2), # torch.nn.ReLU(),
            torch.nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(0.2), # torch.nn.ReLU(),
            torch.nn.Linear(h_dim, 1),
            torch.nn.Sigmoid()
            )

        utils.initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        return x


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
            nn.LeakyReLU(0.2), # torch.nn.ReLU(),
            torch.nn.Linear(h_dim, h_dim),
            torch.nn.BatchNorm1d(h_dim),
            nn.LeakyReLU(0.2), # torch.nn.ReLU(),
            torch.nn.Linear(h_dim, z_dim)
            )

        utils.initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        return x

class Encoder_FC_VAE(nn.Module):
    """
    Simple NN with one hidden layer of dimension h_dim
    Input is vector X from image space if dimension X_dim
    Output is vector z from representation space of dimension z_dim
    """
    def __init__(self, z_dim, h_dim, X_dim):
        super(Encoder_FC_VAE, self).__init__()

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.X_dim = X_dim

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(X_dim, h_dim),
            nn.LeakyReLU(0.2), # torch.nn.ReLU(),
            torch.nn.Linear(h_dim, h_dim),
            torch.nn.BatchNorm1d(h_dim),
            nn.LeakyReLU(0.2), # torch.nn.ReLU(),
            # torch.nn.Linear(h_dim, z_dim)
            )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(h_dim, z_dim)
            )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(h_dim, z_dim)
            )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        z_mu = self.fc1(x)
        z_var = self.fc2(x)
        return z_mu, z_var


class Generator_CNN(nn.Module):
    """
    Simple NN with one hidden layer of dimension h_dim
    Input is a vector from representation space of dimension z_dim
    output is a vector from image space of dimension X_dim
    """
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, z_dim, h_dim, X_dim, params):
        super(Generator_CNN, self).__init__()

        self.input_height = 28
        self.input_width = 28
        self.input_dim = z_dim
        self.output_dim = 1
        self.slope = params['slope']
        self.dropout = params['dropout']
        self.num_channels = self.dropout = params['num_channels']
        self.dataset = params['dataset']



        # self.inference = nn.Sequential(
        #     # input dim: z_dim x 1 x 1
        #     nn.ConvTranspose2d(z_dim, 256, 4, stride=1, bias=True),
        #     nn.BatchNorm2d(256),
        #     nn.LeakyReLU(self.slope, inplace=True),
        #     # state dim:   256 x 4 x 4
        #     nn.ConvTranspose2d(256, 128, 4, stride=2, bias=True),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(self.slope, inplace=True),
        #     # state dim: 128 x 10 x 10
        #     nn.ConvTranspose2d(128, 64, 4, stride=1, bias=True),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(self.slope, inplace=True),
        #     # state dim: 64 x 13 x 13
        #     if self.dataset == 'mnist':
        #         nn.ConvTranspose2d(64, 32, 4, stride=2, bias=True),
        #         nn.BatchNorm2d(32),
        #         nn.LeakyReLU(self.slope, inplace=True),
        #         # state dim: 32 x 28 x 28
        #         nn.Conv2d(32, self.num_channels, 1, stride=1, bias=True),
        #         # output dim: num_channels x 28 x 28
        #     elif self.dataset == 'robot_world':
        #         nn.ConvTranspose2d(64, self.num_channels, 4, stride=1, bias=True),
        #     nn.Tanh()
        # )

        if self.dataset == 'mnist':
            self.inference = nn.Sequential(
                # input dim: z_dim x 1 x 1
                nn.ConvTranspose2d(z_dim, 256, 4, stride=1, bias=True),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(self.slope, inplace=True),
                # state dim:   256 x 4 x 4
                nn.ConvTranspose2d(256, 128, 4, stride=2, bias=True),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(self.slope, inplace=True),
                # state dim: 128 x 10 x 10
                nn.ConvTranspose2d(128, 64, 4, stride=1, bias=True),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(self.slope, inplace=True),
                # state dim: 64 x 13 x 13
                nn.ConvTranspose2d(64, 32, 4, stride=2, bias=True),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(self.slope, inplace=True),
                # state dim: 32 x 28 x 28
                nn.Conv2d(32, self.num_channels, 1, stride=1, bias=True),
                # output dim: num_channels x 28 x 28
                nn.Tanh()
            )
        elif self.dataset == 'robot_world':
            self.inference = nn.Sequential(
                # input dim: z_dim x 1 x 1
                nn.ConvTranspose2d(z_dim, 256, 4, stride=1, bias=True),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(self.slope, inplace=True),
                # state dim:   256 x 4 x 4
                nn.ConvTranspose2d(256, 128, 4, stride=2, bias=True),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(self.slope, inplace=True),
                # state dim: 128 x 10 x 10
                nn.ConvTranspose2d(128, 64, 4, stride=1, bias=True),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(self.slope, inplace=True),
                # state dim: 64 x 13 x 13
                nn.ConvTranspose2d(64, self.num_channels, 4, stride=1, bias=True),
                nn.Sigmoid()
            )

        utils.initialize_weights(self)

    def forward(self, input):
        z = input.unsqueeze(2).unsqueeze(3)
        x = self.inference(z)

        return x

class Encoder_CNN(nn.Module):
    """
    Simple NN with one hidden layer of dimension h_dim
    Input is vector X from image space if dimension X_dim
    Output is vector z from representation space of dimension z_dim
    """
    def __init__(self, z_dim, h_dim, X_dim, params):
        super(Encoder_CNN, self).__init__()

        self.input_height = 28
        self.input_width = 28
        self.input_dim = 1
        self.output_dim = z_dim

        self.slope = params['slope']
        self.dropout = params['dropout']
        self.num_channels = self.dropout = params['num_channels']
        self.dataset = params['dataset']


        # self.inference = nn.Sequential(
        #     if self.dataset == 'mnist':
        #         # input dim: num_channels x 32 x 32
        #         nn.Conv2d(self.num_channels, 32, 3, stride=1, padding=1, bias=True),
        #         nn.BatchNorm2d(32),
        #         nn.LeakyReLU(self.slope, inplace=True),
        #         # state dim: 32 x 28 x 28
        #         nn.Conv2d(32, 64, 4, stride=2, bias=True),
        #         nn.BatchNorm2d(64),
        #         nn.LeakyReLU(self.slope, inplace=True),
        #         # state dim: 64 x 13 x 13
        #     elif self.dataset == 'robot_world':
        #         # input dim: num_channels x 16 x 16
        #         nn.Conv2d(self.num_channels, 64, 4, stride=1, padding=0, bias=True),
        #         # state dim: 64 x 13 x 13
        #     nn.Conv2d(64, 128, 4, stride=1, bias=True),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(self.slope, inplace=True),
        #     # state dim: 128 x 10 x 10
        #     nn.Conv2d(128, 256, 4, stride=2, bias=True),
        #     nn.BatchNorm2d(256),
        #     nn.LeakyReLU(self.slope, inplace=True),
        #     # state dim: 256 x 4 x 4
        #     nn.Conv2d(256, 512, 4, stride=1, bias=True),
        #     nn.BatchNorm2d(512),
        #     nn.LeakyReLU(self.slope, inplace=True),
        #     # state dim: 512 x 1 x 1
        #     nn.Conv2d(512, 512, 1, stride=1, bias=True),
        #     nn.BatchNorm2d(512),
        #     nn.LeakyReLU(self.slope, inplace=True),
        #     # state dim: 512 x 1 x 1
        #     nn.Conv2d(512, z_dim, 1, stride=1, bias=True)
        #     # output dim: opt.z_dim x 1 x 1
        # )

        if self.dataset == 'mnist':
            self.inference = nn.Sequential(
                # input dim: num_channels x 32 x 32
                nn.Conv2d(self.num_channels, 32, 3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(self.slope, inplace=True),
                # state dim: 32 x 28 x 28
                nn.Conv2d(32, 64, 4, stride=2, bias=True),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(self.slope, inplace=True),
                # state dim: 64 x 13 x 13
                nn.Conv2d(64, 128, 4, stride=1, bias=True),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(self.slope, inplace=True),
                # state dim: 128 x 10 x 10
                nn.Conv2d(128, 256, 4, stride=2, bias=True),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(self.slope, inplace=True),
                # state dim: 256 x 4 x 4
                nn.Conv2d(256, 512, 4, stride=1, bias=True),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(self.slope, inplace=True),
                # state dim: 512 x 1 x 1
                nn.Conv2d(512, 512, 1, stride=1, bias=True),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(self.slope, inplace=True),
                # state dim: 512 x 1 x 1
                nn.Conv2d(512, z_dim, 1, stride=1, bias=True)
                # output dim: opt.z_dim x 1 x 1
            )
        elif self.dataset == 'robot_world':
            self.inference = nn.Sequential(
                # input dim: num_channels x 16 x 16
                nn.Conv2d(self.num_channels, 64, 4, stride=1, padding=0, bias=True),
                # state dim: 64 x 13 x 13
                nn.Conv2d(64, 128, 4, stride=1, bias=True),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(self.slope, inplace=True),
                # state dim: 128 x 10 x 10
                nn.Conv2d(128, 256, 4, stride=2, bias=True),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(self.slope, inplace=True),
                # state dim: 256 x 4 x 4
                nn.Conv2d(256, 512, 4, stride=1, bias=True),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(self.slope, inplace=True),
                # state dim: 512 x 1 x 1
                nn.Conv2d(512, 512, 1, stride=1, bias=True),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(self.slope, inplace=True),
                # state dim: 512 x 1 x 1
                nn.Conv2d(512, z_dim, 1, stride=1, bias=True)
                # output dim: opt.z_dim x 1 x 1
            )

        utils.initialize_weights(self)

    def forward(self, input):
        x = self.inference(input)

        return x

class Discriminator_CNN(nn.Module):
    """
    Simple NN with one hidden layer of dimension h_dim
    Input is concatenated vector (X,z) of an image vector and its corresponding
    representation z vector. For example, if X comes from the dataset, corresponding
    z is Encoder(X), and if z is sampled from representation space, X is Generator(z)
    """
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, z_dim, h_dim, X_dim, params):
        super(Discriminator_CNN, self).__init__()

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.X_dim = X_dim

        self.input_height = 28

        self.slope = params['slope']
        self.dropout = params['dropout']
        self.batch_size = params['batch_size']
        self.num_channels = params['num_channels']
        self.dataset = params['dataset']

        if self.dataset == 'mnist':
            self.inference_x = nn.Sequential(
                # state dim: num_channels 28 x 28
                nn.Conv2d(self.num_channels, 64, 4, stride=1, bias=True),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(self.slope, inplace=True),
                nn.Dropout2d(p=self.dropout),
                # state dim: 64 x 13 x 13
                nn.Conv2d(64, 128, 4, stride=1, bias=True),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(self.slope, inplace=True),
                nn.Dropout2d(p=self.dropout),
                # state dim: 128 x 10 x 10
                nn.Conv2d(128, 256, 4, stride=2, bias=True),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(self.slope, inplace=True),
                nn.Dropout2d(p=self.dropout),
                # state dim: 256 x 4 x 4
                nn.Conv2d(256, 512, 4, stride=1, bias=True),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(self.slope, inplace=True),
                nn.Dropout2d(p=self.dropout)
                # output dim: 512 x 1 x 1
            )
        elif self.dataset == 'robot_world':
            self.inference_x = nn.Sequential(
                # state dim: num_channels x 16 x 16
                nn.Conv2d(self.num_channels, 64, 4, stride=1, bias=True),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(self.slope, inplace=True),
                nn.Dropout2d(p=self.dropout),
                # state dim: 64 x 13 x 13
                nn.Conv2d(64, 128, 4, stride=1, bias=True),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(self.slope, inplace=True),
                nn.Dropout2d(p=self.dropout),
                # state dim: 128 x 10 x 10
                nn.Conv2d(128, 256, 4, stride=2, bias=True),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(self.slope, inplace=True),
                nn.Dropout2d(p=self.dropout),
                # state dim: 256 x 4 x 4
                nn.Conv2d(256, 512, 4, stride=1, bias=True),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(self.slope, inplace=True),
                nn.Dropout2d(p=self.dropout)
                # output dim: 512 x 1 x 1
            )

        self.inference_joint = nn.Sequential(
            torch.nn.Linear(512 + self.z_dim, self.h_dim),
            nn.LeakyReLU(0.2), # torch.nn.ReLU(),
            torch.nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU(0.2), # torch.nn.ReLU(),
            torch.nn.Linear(self.h_dim, 1),
            torch.nn.Sigmoid()
        )
        utils.initialize_weights(self)

    def forward(self, x, z):
        output_x = self.inference_x(x)
        output_x = output_x.view(self.batch_size, -1)

        output_z = z.view(self.batch_size, -1)

        output = self.inference_joint(torch.cat((output_x, output_z), 1))
        return output

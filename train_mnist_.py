'''
This is a PyTorch implementation of a BIGAN Network described in the paper
"Adversarial Feature Learning" by J. Donahue, P. Krahenbuhl, T. Darrell.

This implementation is based on [...]

This program will be tested on datasets from "Learning State
Representations with Robotic Priors" (Jonschkowski & Brock, 2015),
https://github.com/tu-rbo/learning-state-representations-with-robotic-priors


'''

import torch, time, os, pickle
import utils
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from itertools import *
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def log(x):
      return torch.log(x + 1e-8)



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
            torch.nn.Sigmoid()
            )

        utils.initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        return x


class Discriminator(nn.Module):
    """
    Simple NN with one hidden layer of dimension h_dim
    Input is concatenated vector (X,z) of an image vector and its corresponding
    representation z vector. For example, if X comes from the dataset, corresponding
    z is Encoder(X), and if z is sampled from representation space, X is Generator(z)
    """
    def __init__(self, z_dim, h_dim, X_dim):
        super(Discriminator, self).__init__()

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.X_dim = X_dim

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(X_dim + z_dim, h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim, h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim, 1),
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

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(X_dim, h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim, h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim, z_dim)
            )
        
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        return x


class BIGAN(object):
    """
    Class implementing a BIGAN network that trains from an observations dataset

    """

    def __init__(self, epoch, sample_num, batch_size, save_dir, result_dir, log_dir, gpu_mode, learning_rate):
        # parameters
        self.epoch = epoch
        self.sample_num = 64
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.gpu_mode = gpu_mode
        self.learning_rate = learning_rate

        self.cnt = 0

        # BIGAN parameters
        self.pt_loss_weight = 0.1
        self.margin = max(1, self.batch_size / 64.)  # margin for loss function
        # usually margin of 1 is enough, but for large batch size it must be larger than 1

        # # load dataset
        # self.data_loader = DataLoader(datasets.MNIST('data/mnist', train=True, download=True,
        #                                              transform=transforms.Compose(
        #                                                  [transforms.ToTensor()])),
        #                               batch_size=self.batch_size, shuffle=True)
        self.mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

        self.z_dim = 10                             #dimension of feature space
        self.h_dim = 1024                            #dimension of the hidden layer
        self.X_dim = self.mnist.train.images.shape[1]    #dimension of data



        # networks init
        self.G = Generator(self.z_dim, self.h_dim, self.X_dim)
        self.D = Discriminator(self.z_dim, self.h_dim, self.X_dim)
        self.E = Encoder(self.z_dim, self.h_dim, self.X_dim)
        # self.l1_reg = l1_reg

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.E.cuda()
            # self.MSE_loss = nn.MSELoss().cuda()
        #else:
            # self.MSE_loss = nn.MSELoss()


        # self.G_solver = optim.Adam(chain(self.E.parameters(), self.G.parameters()), lr=self.learning_rate)
        # self.D_solver = optim.Adam(self.D.parameters(), lr=self.learning_rate)


        self.G_solver = optim.Adam(chain(self.E.parameters(), self.G.parameters()), lr=self.learning_rate, betas=[0.5,0.999], weight_decay=2.5*1e-5)
        self.D_solver = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=[0.5,0.999], weight_decay=2.5*1e-5)



        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.E)
        utils.print_network(self.D)
        print('-----------------------------------------------')



    def D_(self, X, z):
        return self.D(torch.cat([X, z], 1))


    def reset_grad(self):
        self.E.zero_grad()
        self.G.zero_grad()
        self.D.zero_grad()


    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['E_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        print('training start!!')

        for it in range(60000):
            # Sample data
            if self.gpu_mode:
                z = Variable(torch.randn(self.batch_size, self.z_dim)).cuda()
                X, _ = self.mnist.train.next_batch(self.batch_size)
                X = Variable(torch.from_numpy(X)).cuda()
            else:
                z = Variable(torch.randn(self.batch_size, self.z_dim))
                X, _ = self.mnist.train.next_batch(self.batch_size)
                X = Variable(torch.from_numpy(X))

            # Discriminator
            z_hat = self.E(X)
            X_hat = self.G(z)


            D_enc = self.D_(X, z_hat)
            D_gen = self.D_(X_hat, z)

            D_loss = -torch.mean(log(D_enc) + log(1 - D_gen))

            D_loss.backward()
            self.D_solver.step()
            self.G_solver.step()
            self.reset_grad()

            # AutoEncoder Q, P
            z_hat = self.E(X)
            X_hat = self.G(z)

            D_enc = self.D_(X, z_hat)
            D_gen = self.D_(X_hat, z)

            G_loss = -torch.mean(log(D_gen) + log(1 - D_enc))

            G_loss.backward()
            self.G_solver.step()
            self.reset_grad()

            # Print and plot every now and then
            if it % 1000 == 0:
                print('Iter-{}; D_loss: {:.4}; G_loss: {:.4}'
                      .format(it, D_loss.data[0], G_loss.data[0]))

                samples = self.G(z).data.cpu().numpy()[:16]

                fig = plt.figure(figsize=(4, 4))
                gs = gridspec.GridSpec(4, 4)
                gs.update(wspace=0.05, hspace=0.05)

                for i, sample in enumerate(samples):
                    ax = plt.subplot(gs[i])
                    plt.axis('off')
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_aspect('equal')
                    plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

                if not os.path.exists('out/'):
                    os.makedirs('out/')

                plt.savefig('out/{}.png'.format(str(self.cnt).zfill(3)), bbox_inches='tight')
                self.cnt += 1
        plt.close(fig)


if __name__ == '__main__':
    m_BIGAN = BIGAN(epoch = 2, sample_num = 2, batch_size = 128, save_dir = "save_dir",
        result_dir = "result_dir", log_dir = "log_dir", gpu_mode = True, learning_rate = 1e-3)
    m_BIGAN.train()

    if not os.path.exists('saved_model/'):
        os.makedirs('saved_model/')

    torch.save(m_BIGAN.G.state_dict(), "saved_model/G.pt")
    torch.save(m_BIGAN.E.state_dict(), "saved_model/E.pt")
    torch.save(m_BIGAN.D.state_dict(), "saved_model/D.pt")
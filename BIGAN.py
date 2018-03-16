import numpy as np

import torch, time, os, pickle
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import datasets, transforms
import utils
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from itertools import *
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import math
from functools import reduce

from plot_utils import save_plot_losses, save_plot_pixel_norm
from models import Generator_FC, Discriminator_FC, Encoder_FC, Generator_CNN, Discriminator_CNN, Encoder_CNN

USE_CUDA = True
EPS = 1e-12
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

def log(x):
      return torch.log(x + 1e-8)

class Mnist:
    def __init__(self, batch_size):
        dataset_transform = transforms.Compose([
                       transforms.ToTensor(),
                       # transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
                   ])

        train_dataset = datasets.MNIST('../data', train=True, download=True, transform=dataset_transform)
        test_dataset = datasets.MNIST('../data', train=False, download=True, transform=dataset_transform)

        self.train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

class RobotWorld:
    def __init__(self, batch_size, dataset_path, gpu_mode):
        self.gpu_mode = gpu_mode
        self.batch_size = batch_size

        print('Loading data ... ')

        ###########################################################
        path = dataset_path + '/simple_navigation_task_train.npz'
        training_data = np.load(path)

        observations, actions = training_data['observations'], training_data['actions']
        rewards, episode_starts = training_data['rewards'], training_data['episode_starts']
        obs_dim = reduce(lambda x,y: x*y, observations.shape[1:])

        if len(observations.shape) > 2:
            # Channel first
            observations = np.transpose(observations, (0, 3, 1, 2))
            # Flatten the image
            observations = observations.reshape((-1, obs_dim))

        ###########################################################
        path = dataset_path + '/simple_navigation_task_test.npz'
        test_data = np.load(path)

        test_observations, actions = test_data['observations'], test_data['actions']
        rewards, episode_starts = test_data['rewards'], test_data['episode_starts']
        test_obs_dim = reduce(lambda x,y: x*y, test_observations.shape[1:])

        if len(test_observations.shape) > 2:
            # Channel first
            test_observations = np.transpose(test_observations, (0, 3, 1, 2))
            # Flatten the image
            test_observations = test_observations.reshape((-1, test_obs_dim))

        ###########################################################
        self.observations = observations.astype(np.float32)

        obs_var = Variable(torch.from_numpy(observations), volatile=True)
        if self.gpu_mode:
            obs_var = obs_var.cuda()

        num_samples = observations.shape[0] - 1 # number of samples

        # indices for all time steps where the episode continues
        indices = np.array([i for i in range(num_samples)], dtype='int64')
        np.random.shuffle(indices)

        # split indices into minibatches
        self.minibatchlist = [np.array(sorted(indices[start_idx:start_idx + self.batch_size]))
            for start_idx in range(0, num_samples - self.batch_size + 1, self.batch_size)]

        ###########################################################

        self.test_observations = test_observations.astype(np.float32)

        test_obs_var = Variable(torch.from_numpy(test_observations), volatile=True)
        if self.gpu_mode:
            test_obs_var = test_obs_var.cuda()

        num_test_samples = test_observations.shape[0] - 1 # number of samples

        # indices for all time steps where the episode continues
        indices = np.array([i for i in range(num_test_samples)], dtype='int64')
        np.random.shuffle(indices)

        # split indices into minibatches
        self.test_minibatchlist = [np.array(sorted(indices[start_idx:start_idx + self.batch_size]))
            for start_idx in range(0, num_test_samples - self.batch_size + 1, self.batch_size)]

        ###########################################################

        self.train_loader = [ ( torch.from_numpy(self.observations[batch]).float() , 0)
            for it, batch in list(enumerate(self.minibatchlist)) ]
        self.test_loader =  [ ( torch.from_numpy(self.test_observations[batch]).float() , 0)
            for it, batch in list(enumerate(self.test_minibatchlist)) ]

    def shuffle(self):
        enumerated_minibatches = list(enumerate(self.minibatchlist))
        np.random.shuffle(enumerated_minibatches)

        enumerated_test_minibatches = list(enumerate(self.test_minibatchlist))
        np.random.shuffle(enumerated_test_minibatches)

        self.train_loader = [ ( torch.from_numpy(self.observations[batch]).float() , 0)
            for it, batch in list(enumerate(self.minibatchlist)) ]
        self.test_loader =  [ ( torch.from_numpy(self.test_observations[batch]).float() , 0)
            for it, batch in list(enumerate(self.test_minibatchlist)) ]

class BIGAN(object):
    """
    Class implementing a BIGAN network that trains from an observations dataset
    """

    def __init__(self, args):
    # def __init__(self, epoch, batch_size, save_dir, result_dir, log_dir, gpu_mode, learning_rate):
        # parameters
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.learning_rate = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.slope = args.slope
        self.decay = args.decay
        self.dropout = args.dropout
        self.network_type = args.network_type
        self.dataset = args.dataset
        self.dataset_path = args.dataset_path

        # BIGAN parameters
        self.z_dim = args.z_dim     #dimension of feature space
        self.h_dim = args.h_dim     #dimension of the hidden layer

        if args.dataset == 'mnist':
            self.X_dim = 28*28                                 #dimension of data
            self.num_channels = 1
        elif args.dataset == 'robot_world':
            self.X_dim = 16*16*3                                 #dimension of data
            self.num_channels = 3

        if args.network_type == 'FC':
            # networks init
            self.G = Generator_FC(self.z_dim, self.h_dim, self.X_dim)
            self.D = Discriminator_FC(self.z_dim, self.h_dim, self.X_dim)
            self.E = Encoder_FC(self.z_dim, self.h_dim, self.X_dim)
        elif args.network_type == 'CNN':
            params = {'slope': self.slope, 'dropout':self.dropout, 'batch_size':self.batch_size, 'num_channels':self.num_channels}

            self.G = Generator_CNN(self.z_dim, self.h_dim, self.X_dim, params)
            self.D = Discriminator_CNN(self.z_dim, self.h_dim, self.X_dim, params)
            self.E = Encoder_CNN(self.z_dim, self.h_dim, self.X_dim, params)
        else:
            raise Exception("[!] There is no option for " + args.network_type)

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.E.cuda()

        self.G_solver = optim.Adam(chain(self.E.parameters(), self.G.parameters()), lr=self.learning_rate, betas=[self.beta1,self.beta2], weight_decay=self.decay)
        self.D_solver = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=[self.beta1,self.beta2], weight_decay=self.decay)



        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.E)
        utils.print_network(self.D)
        print('-----------------------------------------------')



    def D_(self, X, z):
        return self.D(torch.cat([X, z], 1))

    def D_2(self, X, z):
        # Sample data
        if self.gpu_mode:
            X = X.data.cpu().numpy()
            X = X.reshape(self.batch_size, -1)
            X = Variable(torch.from_numpy(X)).cuda()
        else:
            X = X.data.cpu().numpy()
            X = X.reshape(self.batch_size, -1)
            X = Variable(torch.from_numpy(X))

        return self.D(torch.cat([X, z], 1))


    def reset_grad(self):
        self.E.zero_grad()
        self.G.zero_grad()
        self.D.zero_grad()


    def train(self):
        if self.dataset == 'mnist':
            dataset = Mnist(self.batch_size)
        elif self.dataset == 'robot_world':
            dataset = RobotWorld(self.batch_size, self.dataset_path, self.gpu_mode)


        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []

        self.eval_hist = {}
        self.eval_hist['D_loss'] = []
        self.eval_hist['G_loss'] = []
        self.eval_hist['pixel_norm'] = []


        for epoch in range(self.epoch):
            print("epoch ",str(epoch))

            self.D.train()
            self.E.train()
            self.G.train()

            train_loss_G = 0
            train_loss_D = 0

            if self.dataset == "robot_world":
                dataset.shuffle()

            for batch_id, (data, target) in enumerate(dataset.train_loader):

                if self.gpu_mode:
                    z = Variable(torch.rand(self.batch_size, self.z_dim)).cuda()
                    X = data
                    X = Variable(X).cuda()
                else:
                    z = Variable(torch.rand(self.batch_size, self.z_dim))
                    X = data
                    X = Variable(X)

                if X.size(0) == self.batch_size:

                    if self.network_type == 'CNN':
                        z_hat = self.E(X)
                        X_hat = self.G(z)

                        D_enc = self.D(X, z_hat)
                        z = z.unsqueeze(2).unsqueeze(3)
                        D_gen = self.D(X_hat, z)

                    elif self.network_type == 'FC':
                        X = X.view(self.batch_size, -1)
                        z_hat = self.E(X)
                        X_hat = self.G(z)

                        D_enc = self.D_(X, z_hat)
                        D_gen = self.D_(X_hat, z)

                    D_loss = -torch.mean(log(D_enc) + log(1 - D_gen))
                    G_loss = -torch.mean(log(D_gen) + log(1 - D_enc))

                    D_loss.backward(retain_graph=True)
                    self.D_solver.step()
                    self.reset_grad()

                    G_loss.backward()
                    self.G_solver.step()
                    self.reset_grad()

                    train_loss_G += G_loss.data[0]
                    train_loss_D += D_loss.data[0]

                    if batch_id % 1000 == 0:
                        # Print and plot every now and then
                        samples = X_hat.data.cpu().numpy()

                        fig = plt.figure(figsize=(8, 4))
                        gs = gridspec.GridSpec(4, 8)
                        gs.update(wspace=0.05, hspace=0.05)

                        for i, sample in enumerate(samples):
                            if i<32:
                                ax = plt.subplot(gs[i])
                                plt.axis('off')
                                ax.set_xticklabels([])
                                ax.set_yticklabels([])
                                ax.set_aspect('equal')

                                if self.network_type == 'FC':
                                    if self.dataset == 'mnist':
                                        sample = sample.reshape(28, 28)
                                        # sample = sample*MNIST_STD + MNIST_MEAN
                                        plt.imshow(sample, cmap='Greys_r')
                                    elif self.dataset == 'robot_world':
                                        sample = sample.reshape(16,16,3)
                                        sample = np.rot90(sample, 2)
                                        plt.imshow(sample)
                                elif self.network_type == 'CNN':
                                    if self.dataset == 'mnist':
                                        # sample = sample*MNIST_STD + MNIST_MEAN
                                        plt.imshow(sample[0,:,:], cmap='Greys_r')
                                    elif self.dataset == 'robot_world':
                                        sample = sample.reshape(16,16,3)
                                        sample = np.rot90(sample, 2)
                                        plt.imshow(sample)


                        if not os.path.exists(self.result_dir + '/train/'):
                            os.makedirs(self.result_dir + '/train/')

                        filename = "epoch_" + str(epoch) + "_batchid_" + str(batch_id)
                        plt.savefig(self.result_dir + '/train/{}.png'.format(filename, bbox_inches='tight'))

            print("Train loss G:", train_loss_G / len(dataset.train_loader))
            print("Train loss D:", train_loss_D / len(dataset.train_loader))

            self.train_hist['D_loss'].append(train_loss_D / len(dataset.train_loader))
            self.train_hist['G_loss'].append(train_loss_G / len(dataset.train_loader))


            self.D.eval()
            self.E.eval()
            self.G.eval()
            test_loss_G = 0
            test_loss_D = 0

            mean_pixel_norm = 0

            for batch_id, (data, target) in enumerate(dataset.test_loader):

                # Sample data
                if self.gpu_mode:
                    z = Variable(torch.randn(self.batch_size, self.z_dim)).cuda()
                    X_data = data
                    X_data = Variable(X_data).cuda()
                else:
                    z = Variable(torch.randn(self.batch_size, self.z_dim))
                    X_data = data
                    X_data = Variable(X_data)


                if X_data.size(0) == self.batch_size:
                    if self.network_type == 'CNN':
                        z_hat = self.E(X)
                        X_hat = self.G(z)

                        D_enc = self.D(X, z_hat)
                        z = z.unsqueeze(2).unsqueeze(3)
                        D_gen = self.D(X_hat, z)

                    elif self.network_type == 'FC':
                        X = X.view(self.batch_size, -1)
                        z_hat = self.E(X)
                        X_hat = self.G(z)

                        D_enc = self.D_(X, z_hat)
                        D_gen = self.D_(X_hat, z)

                    D_loss = -torch.mean(log(D_enc) + log(1 - D_gen))
                    G_loss = -torch.mean(log(D_gen) + log(1 - D_enc))

                    test_loss_G += G_loss.data[0]
                    test_loss_D += D_loss.data[0]

                    pixel_norm = X - X_hat
                    pixel_norm = pixel_norm = pixel_norm.norm()

                    mean_pixel_norm += int(pixel_norm)


            print("Eval loss G:", test_loss_G / len(dataset.test_loader))
            print("Eval loss D:", test_loss_D / len(dataset.test_loader))

            self.eval_hist['D_loss'].append(test_loss_D / len(dataset.test_loader))
            self.eval_hist['G_loss'].append(test_loss_G / len(dataset.test_loader))

            print("Pixel norm:", mean_pixel_norm / len(dataset.test_loader))
            self.eval_hist['pixel_norm'].append( mean_pixel_norm / len(dataset.test_loader) )

            with open('pixel_error_BIGAN.txt', 'a') as f:
                f.writelines(str(mean_pixel_norm / len(dataset.test_loader)) + '\n')

            mean_pixel_norm = 0

            ##### save X and G(E(X))
            samples = X.data.cpu().numpy() #samples = X.data.cpu().numpy()[:32]

            fig = plt.figure(figsize=(8, 4))
            gs = gridspec.GridSpec(4, 8)
            gs.update(wspace=0.05, hspace=0.05)

            for i, sample in enumerate(samples):
                if i<32:
                    ax = plt.subplot(gs[i])
                    plt.axis('off')
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_aspect('equal')
                    if self.network_type == 'FC':
                        if self.dataset == 'mnist':
                            sample = sample.reshape(28, 28)
                            # sample = sample*MNIST_STD + MNIST_MEAN
                            plt.imshow(sample, cmap='Greys_r')
                        elif self.dataset == 'robot_world':
                            sample = sample.reshape(16,16,3)
                            sample = np.rot90(sample, 2)
                            plt.imshow(sample)
                    elif self.network_type == 'CNN':
                        if self.dataset == 'mnist':
                            # sample = sample*MNIST_STD + MNIST_MEAN
                            plt.imshow(sample[0,:,:], cmap='Greys_r')
                        elif self.dataset == 'robot_world':
                            sample = sample.reshape(16,16,3)
                            sample = np.rot90(sample, 2)
                            plt.imshow(sample)

            if not os.path.exists(self.result_dir + '/real/'):
                os.makedirs(self.result_dir + '/real/')

            filename = "epoch_" + str(epoch)
            plt.savefig(self.result_dir + '/real/{}.png'.format(filename), bbox_inches='tight')
            plt.close(fig)

            X_hat = self.G(self.E(X).view(self.batch_size, self.z_dim))
            samples = X_hat.data.cpu().numpy() #samples = X_hat.data.cpu().numpy()[:32]

            fig = plt.figure(figsize=(8, 4))
            gs = gridspec.GridSpec(4, 8)
            gs.update(wspace=0.05, hspace=0.05)

            for i, sample in enumerate(samples):
                if i<32:
                    ax = plt.subplot(gs[i])
                    plt.axis('off')
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_aspect('equal')
                    if self.network_type == 'FC':
                        if self.dataset == 'mnist':
                            sample = sample.reshape(28, 28)
                            # sample = sample*MNIST_STD + MNIST_MEAN
                            plt.imshow(sample, cmap='Greys_r')
                        elif self.dataset == 'robot_world':
                            sample = sample.reshape(16,16,3)
                            sample = np.rot90(sample, 2)
                            plt.imshow(sample)
                    elif self.network_type == 'CNN':
                        if self.dataset == 'mnist':
                            # sample = sample*MNIST_STD + MNIST_MEAN
                            plt.imshow(sample[0,:,:], cmap='Greys_r')
                        elif self.dataset == 'robot_world':
                            sample = sample.reshape(16,16,3)
                            sample = np.rot90(sample, 2)
                            plt.imshow(sample)

            if not os.path.exists(self.result_dir + '/recons/'):
                os.makedirs(self.result_dir + '/recons/')

            filename = "epoch_" + str(epoch)
            plt.savefig(self.result_dir + '/recons/{}.png'.format(filename), bbox_inches='tight')
            plt.close(fig)

        save_plot_losses(self.train_hist['D_loss'], self.train_hist['G_loss'], self.eval_hist['D_loss'], self.eval_hist['G_loss'])
        save_plot_pixel_norm(self.eval_hist['pixel_norm'])

    def save_model(self):
        torch.save(self.G.state_dict(), self.save_dir + "/G.pt")
        torch.save(self.E.state_dict(), self.save_dir + "/E.pt")
        torch.save(self.D.state_dict(), self.save_dir + "/D.pt")

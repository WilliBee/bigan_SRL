# THIS IS AN ALTERATION OF THE ORIGINAL CODE TO TEST VAE !!
# WORKS ONLY IN THE FULLY CONNECTED MODE




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

from plot_utils import save_plot_losses, save_plot_pixel_norm, save_plot_z_norm

from model import Generator, Encoder

from representation_plot import plot_representation, plot_representation2

import matplotlib.mlab as mlab

def log(x):
      return torch.log(x + 1e-8)

class Mnist:
    def __init__(self, batch_size):
        MNIST_MEAN = 0.1307
        MNIST_STD = 0.3081

        dataset_transform = transforms.Compose([
                       transforms.ToTensor(),
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

        self.train_loader = [ ( torch.from_numpy(self.observations[batch]).float() , it)
            for it, batch in enumerated_minibatches ]
        self.test_loader =  [ ( torch.from_numpy(self.test_observations[batch]).float() , it)
            for it, batch in enumerated_test_minibatches ]

class VAE(object):
    """
    Class implementing a BIGAN network that trains from an observations dataset
    """

    def __init__(self, args):
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
            self.G = Generator(self.z_dim, self.h_dim, self.X_dim)
            self.E = Encoder(self.z_dim, self.h_dim, self.X_dim)
        else:
            raise Exception("[!] There is no option for " + args.network_type)

        if self.gpu_mode:
            self.G.cuda()
            self.E.cuda()

        self.G_solver = optim.Adam(chain(self.E.parameters(), self.G.parameters()), lr=self.learning_rate, betas=[self.beta1,self.beta2], weight_decay=self.decay)



        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.E)
        print('-----------------------------------------------')

    def sample_z(self, mu, log_var):
        # Using reparameterization trick to sample from a gaussian
        eps = Variable(torch.randn(self.batch_size, self.z_dim))

        if self.gpu_mode:
            eps = eps.cuda()

        return mu + torch.exp(log_var / 2) * eps


    def reset_grad(self):
        self.E.zero_grad()
        self.G.zero_grad()


    def train(self):
        if self.dataset == 'mnist':
            dataset = Mnist(self.batch_size)
        elif self.dataset == 'robot_world':
            dataset = RobotWorld(self.batch_size, self.dataset_path, self.gpu_mode)


        self.train_hist = {}
        self.train_hist['G_loss'] = []

        self.eval_hist = {}
        self.eval_hist['G_loss'] = []
        self.eval_hist['pixel_norm'] = []
        self.eval_hist['z_norm'] = []


        for epoch in range(self.epoch):
            print("epoch ",str(epoch))

            self.E.train()
            self.G.train()

            train_loss_G = 0

            if self.dataset == "robot_world":
                dataset.shuffle()

            for batch_id, (data, target) in enumerate(dataset.train_loader):

                if self.gpu_mode:
                    z = Variable(torch.randn(self.batch_size, self.z_dim)).cuda()
                    X = data
                    X = Variable(X).cuda()
                else:
                    z = Variable(torch.randn(self.batch_size, self.z_dim))
                    X = data
                    X = Variable(X)

                if X.size(0) == self.batch_size:

                    if self.network_type == 'FC':
                        X = X.view(self.batch_size, -1)


                    # Forward
                    z_mu, z_var = self.E(X)

                    z = self.sample_z(z_mu, z_var)
                    X_sample = self.G(z)

                    # Loss
                    recon_loss = nn.functional.binary_cross_entropy(X_sample, X, size_average=False)
                    kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var)
                    loss = recon_loss + kl_loss

                    # Backward
                    loss.backward()

                    # Update
                    self.G_solver.step()

                    # Housekeeping
                    self.reset_grad()

                    train_loss_G += loss.data[0]

                    if batch_id % 1000 == 0:
                        # Print and plot every now and then
                        samples = X_sample.data.cpu().numpy()

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
                                        plt.imshow(sample, cmap='Greys_r')
                                    elif self.dataset == 'robot_world':
                                        sample = sample.reshape(16,16,3)
                                        sample = np.rot90(sample, 2)
                                        plt.imshow(sample)


                        if not os.path.exists(self.result_dir + '/train/'):
                            os.makedirs(self.result_dir + '/train/')

                        filename = "epoch_" + str(epoch) + "_batchid_" + str(batch_id)
                        plt.savefig(self.result_dir + '/train/{}.png'.format(filename, bbox_inches='tight'))
                        plt.close()

            print("Train loss G:", train_loss_G / len(dataset.train_loader))

            self.train_hist['G_loss'].append(train_loss_G / len(dataset.train_loader))


            self.E.eval()
            self.G.eval()
            test_loss_G = 0

            mean_pixel_norm = 0
            mean_z_norm = 0
            norm_counter = 1

            for batch_id, (data, target) in enumerate(dataset.test_loader):
                # Sample data
                z = Variable(torch.randn(self.batch_size, self.z_dim))
                X_data = Variable(data)

                if self.gpu_mode:
                    z = z.cuda()
                    X_data = X_data.cuda()

                if X_data.size(0) == self.batch_size:
                    X = X_data

                    if self.network_type == 'FC':
                        X = X.view(self.batch_size, -1)
                    else:
                        print("please select FC as network type")

                    # Forward
                    z_mu, z_var = self.E(X)
                    z = self.sample_z(z_mu, z_var)
                    X_hat = self.G(z)

                    X_hat = X_hat.view(self.batch_size, -1)

                    pixel_norm = X - X_hat
                    pixel_norm = pixel_norm.norm().data[0] / (16*16*3)

                    mean_pixel_norm += pixel_norm

                    z_mu_hat, z_var_hat = self.E(self.G(z))


                    z_norm = z - z_mu_hat
                    z_norm = z_norm.norm().data[0] / (self.z_dim)

                    mean_z_norm += z_norm

                    norm_counter += 1


            print("Eval loss G:", test_loss_G / norm_counter)

            self.eval_hist['G_loss'].append(test_loss_G / norm_counter)

            print("Pixel norm:", mean_pixel_norm / norm_counter)
            self.eval_hist['pixel_norm'].append( mean_pixel_norm / norm_counter )

            with open('pixel_error_BIGAN.txt', 'a') as f:
                f.writelines(str(mean_pixel_norm / norm_counter) + '\n')

            print("z norm:", mean_z_norm / norm_counter)
            self.eval_hist['z_norm'].append( mean_z_norm / norm_counter )

            with open('z_error_BIGAN.txt', 'a') as f:
                f.writelines(str(mean_z_norm / norm_counter) + '\n')

            ##### save X and G(E(X))
            samples = X.data.cpu().numpy()

            fig = plt.figure(figsize=(10, 2))
            gs = gridspec.GridSpec(2, 10)
            gs.update(wspace=0.05, hspace=0.05)

            for i, sample in enumerate(samples):
                if i<10:
                    ax = plt.subplot(gs[i])
                    plt.axis('off')
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_aspect('equal')
                    if self.network_type == 'FC':
                        if self.dataset == 'mnist':
                            sample = sample.reshape(28, 28)
                            plt.imshow(sample, cmap='Greys_r')
                        elif self.dataset == 'robot_world':
                            sample = sample.reshape(16,16,3)
                            sample = np.rot90(sample, 2)
                            plt.imshow(sample)

            samples = X_hat.data.cpu().numpy()


            for i, sample in enumerate(samples):
                if i<10:
                    ax = plt.subplot(gs[10+i])
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


            if not os.path.exists(self.result_dir + '/recons/'):
                os.makedirs(self.result_dir + '/recons/')

            filename = "epoch_" + str(epoch)
            plt.savefig(self.result_dir + '/recons/{}.png'.format(filename), bbox_inches='tight')
            plt.close()

            if epoch % 10 == 0:
                self.plot_states(epoch)

        save_plot_losses(self.train_hist['G_loss'], self.eval_hist['D_loss'], self.eval_hist['G_loss'], self.network_type, self.z_dim, self.epoch, self.learning_rate, self.batch_size)
        save_plot_pixel_norm(self.eval_hist['pixel_norm'], self.network_type, self.z_dim, self.epoch, self.learning_rate, self.batch_size)
        save_plot_z_norm(self.eval_hist['z_norm'], self.network_type, self.z_dim, self.epoch, self.learning_rate, self.batch_size)

    def save_model(self):
        torch.save(self.G.state_dict(), self.save_dir + "/G.pt")
        torch.save(self.E.state_dict(), self.save_dir + "/E.pt")

    def load_model(self, args):
        if args.network_type == 'FC':
            # networks init
            self.G = Generator_FC(self.z_dim, self.h_dim, self.X_dim)
            self.E = Encoder_FC(self.z_dim, self.h_dim, self.X_dim)


        self.G.load_state_dict(torch.load("models/G.pt"))
        self.E.load_state_dict(torch.load("models/E.pt"))

        if self.gpu_mode:
            self.G.cuda()
            self.E.cuda()

    def plot_states(self, i):
        if self.dataset == 'robot_world':
            test_data = np.load(self.dataset_path + '/simple_navigation_task_test.npz')

            test_observations, actions = test_data['observations'], test_data['actions']
            rewards, episode_starts = test_data['rewards'], test_data['episode_starts']
            test_obs_dim = reduce(lambda x,y: x*y, test_observations.shape[1:])

            if len(test_observations.shape) > 2:
                # Channel first
                test_observations = np.transpose(test_observations, (0, 3, 1, 2))
                # Flatten the image
                test_observations = test_observations.reshape((-1, test_obs_dim))


            test_observations = test_observations.astype(np.float32)

            obs_var = Variable(torch.from_numpy(test_observations), volatile=True)
            if self.gpu_mode:
                obs_var = obs_var.cuda()

            num_samples = test_observations.shape[0] - 1

            print("NUM SAMPLES IS " + str(num_samples))

            # indices for all time steps where the episode continues
            indices = np.array([i for i in range(num_samples)], dtype='int64')

            # split indices into minibatches
            minibatchlist = [np.array(sorted(indices[start_idx:start_idx + self.batch_size]))
                for start_idx in range(0, num_samples - self.batch_size + 1, self.batch_size)]

            enumerated_minibatches = list(enumerate(minibatchlist))

            for it, batch in enumerated_minibatches:
                obs = Variable(torch.from_numpy(test_observations[batch]).float())

                # Sample data
                if self.gpu_mode:
                    X = obs.cuda()
                else:
                    X = batch

                z_mu, z_var = self.E(X)
                z_hat = z_mu

                ############

                if it==0:
                    states = z_hat.data.cpu().numpy()
                else:
                    states = np.vstack((states , z_hat.data.cpu().numpy() ))


            rewards = test_data['rewards']
            rewards = rewards[:len(states)]

            print("LEN OF REWARDS IS : ", len(rewards))
            print('LEN OF STATES IS : ', len(states))

            if self.z_dim == 2:
                plot_representation2(states, rewards, self.network_type, self.z_dim, self.epoch, self.learning_rate, self.batch_size, i)
            else:
                plot_representation(states, rewards, self.network_type, self.z_dim, self.epoch, self.learning_rate, self.batch_size, i)
                plot_representation2(states, rewards, self.network_type, self.z_dim, self.epoch, self.learning_rate, self.batch_size, i)


def plot_representation_2D(states, rewards, model_used, z_dim, epochs, lr, batch_size, name="Learned State Representation", add_colorbar=True):
    plt.ion()

    fig = plt.figure()
    ax = fig.add_subplot(111, axisbg = 'grey')

    ax.scatter(states[:, 0], states[:, 1], s=7, c=np.clip(rewards, -1, 1), cmap='bwr', linewidths=0.1, label=name)
    plt.xlabel('State dimension 1')
    plt.ylabel('State dimension 2')

    ax.set_aspect('equal',  adjustable='box')

    params = "Network type: " + model_used + ", Dimension of latent space: " + str(z_dim) + ", epochs: " + str(epochs) + ", learning rate: " + str(lr) + ", batch size:" + str(batch_size)
    plt.title(params, fontsize=8)

    plt.savefig("representation_plot.eps", format='eps', dpi=1000)
    plt.close()

def plot_z_distribution(z, model_used, z_dim, epochs, lr, batch_size):
    if not os.path.exists('histograms'):
        os.makedirs('histograms')



    for i in range(z.shape[1]):
        fig = plt.figure()
        # the histogram of the data
        n, bins, patches = plt.hist(z[:,i], 50, normed=1, facecolor='orange', alpha=0.75)

        plt.xlabel('z_' + str(i))
        plt.ylabel('Probability')
        plt.suptitle(r'Histogram of z distribution in dim ' + str(i))
        params = "Network type: " + model_used + ", Dimension of latent space: " + str(z_dim) + ", epochs: " + str(epochs) + ", learning rate: " + str(lr) + ", batch size:" + str(batch_size)
        plt.title(params, fontsize=8)

        plt.grid(True)

        plt.savefig("histograms/histogram_z_" + str(i) + ".eps", format='eps', dpi=1000)
        plt.close()

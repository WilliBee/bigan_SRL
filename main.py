"""
Testing BIGAN (Adversarial Feature Learning) for State Representation Learning

This is a PyTorch implementation of a BIGAN Network described in the paper "Adversarial Feature Learning" by J. Donahue, P. Krahenbuhl, T. Darrell.

This program will be tested on datasets from "Learning State Representations with Robotic Priors" (Jonschkowski & Brock, 2015), https://github.com/tu-rbo/learning-state-representations-with-robotic-priors


"""


import argparse, os
from BIGAN import BIGAN

"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of BIGAN"
    parser = argparse.ArgumentParser(description=desc)


    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'robot_world'], help='The name of dataset')
    parser.add_argument('--dataset_path', type=str, default='/home/williamb/project/bigan_implementation/m_bigan_learner/racecar_dataset')
    parser.add_argument('--gpu_mode', type=bool, default=True)

    # logging
    parser.add_argument('--save_dir', type=str, default='models', help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')

    # hyperparameters
    parser.add_argument('--epoch', type=int, default=25, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='for adam')
    parser.add_argument('--slope', type=float, default=1e-2, help='for leaky ReLU')
    parser.add_argument('--decay', type=float, default=2.5*1e-5, help='for weight decay')
    parser.add_argument('--dropout', type=float, default=0.2)

    # network parameters
    parser.add_argument('--network_type', type=str, default='FC', choices=['FC', 'CNN'], help='Type of network (Fully connectec or CNN)')
    parser.add_argument('--z_dim', type=int, default=50, help='The dimension of latent space Z')
    parser.add_argument('--h_dim', type=int, default=1024, help='The dimension of the hidden layers in case of a FC network')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    bigan = BIGAN(args)

    # ecrase anciens fichiers
    with open('pixel_error_BIGAN.txt', 'w') as f:
        f.writelines('')
    with open('z_error_BIGAN.txt', 'w') as f:
        f.writelines('')

    bigan.train()
    print(" [*] Training finished!")

    bigan.save_model()


    bigan.plot_states()

if __name__ == '__main__':
    main()

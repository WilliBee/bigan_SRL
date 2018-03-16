import matplotlib.pyplot as plt
import numpy as np

def save_plot_losses(train_D_loss, train_G_loss, eval_D_loss, eval_G_loss):

    x = np.arange(1, len(train_D_loss) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(x, train_G_loss, label="Train G loss", linewidth=2)
    plt.plot(x, eval_G_loss, label="Eval G loss", linewidth=2)


    plt.axes().set_xlabel('Epoch')
    plt.axes().set_ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title("Evolution of the Train and Eval losses of G")
    # plt.show()
    plt.savefig("plot_G_losses.eps", format='eps', dpi=1000)



    plt.figure(figsize=(8, 6))
    plt.plot(x, train_D_loss, label="Train D loss", linewidth=2)
    plt.plot(x, eval_D_loss, label="Eval D loss", linewidth=2)

    plt.axes().set_xlabel('Epoch')
    plt.axes().set_ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title("Evolution of the Train and Eval losses of D")
    # plt.show()
    plt.savefig("plot_D_losses.eps", format='eps', dpi=1000)



def save_plot_pixel_norm(mean_pixel_norm):
    x = np.arange(1, len(mean_pixel_norm) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(x, mean_pixel_norm, label="Reconstruction error", linewidth=2)

    plt.axes().set_xlabel('Epoch')
    plt.axes().set_ylabel('Norm')
    plt.legend(loc='upper right')
    plt.title("Evolution of the reconstruction error between X and G(E(X))")
    # plt.show()
    plt.savefig("pix2pix_norm.eps", format='eps', dpi=1000)

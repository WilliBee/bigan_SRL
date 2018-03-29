import matplotlib.pyplot as plt
import numpy as np

def save_plot_losses(train_G_loss, eval_D_loss, eval_G_loss, model_used, z_dim, epochs, lr, batch_size):

    x = np.arange(1, len(train_D_loss) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(x, train_G_loss, label="Train G loss", linewidth=2)
    plt.plot(x, eval_G_loss, label="Eval G loss", linewidth=2)


    plt.axes().set_xlabel('Epoch')
    plt.axes().set_ylabel('Loss')
    plt.legend(loc='upper right')
    plt.suptitle("Evolution of the Train and Eval losses of G")
    params = "Network type: " + model_used + ", Dimension of latent space: " + str(z_dim) + ", epochs: " + str(epochs) + ", learning rate: " + str(lr) + ", batch size:" + str(batch_size)
    plt.title(params, fontsize=8)
    # plt.show()
    plt.savefig("plot_G_losses.eps", format='eps', dpi=1000)



    # plt.figure(figsize=(8, 6))
    # plt.plot(x, train_D_loss, label="Train D loss", linewidth=2)
    # plt.plot(x, eval_D_loss, label="Eval D loss", linewidth=2)
    #
    # plt.axes().set_xlabel('Epoch')
    # plt.axes().set_ylabel('Loss')
    # plt.legend(loc='upper right')
    #
    # plt.suptitle("Evolution of the Train and Eval losses of D")
    #
    # params = "Network type: " + model_used + ", Dimension of latent space: " + str(z_dim) + ", epochs: " + str(epochs) + ", learning rate: " + str(lr) + ", batch size:" + str(batch_size)
    # plt.title(params, fontsize=8)
    # # plt.show()
    # plt.savefig("plot_D_losses.eps", format='eps', dpi=1000)
    plt.close()


def save_plot_pixel_norm(mean_pixel_norm, model_used, z_dim, epochs, lr, batch_size):
    x = np.arange(1, len(mean_pixel_norm) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(x, mean_pixel_norm, label="Reconstruction error", linewidth=2)

    plt.axes().set_xlabel('Epoch')
    plt.axes().set_ylabel('Norm')
    plt.legend(loc='upper right')
    plt.suptitle("Evolution of the reconstruction error between X and G(E(X))")
    params = "Network type: " + model_used + ", Dimension of latent space: " + str(z_dim) + ", epochs: " + str(epochs) + ", learning rate: " + str(lr) + ", batch size:" + str(batch_size)
    plt.title(params, fontsize=8)
    # plt.show()
    plt.savefig("pix2pix_norm.eps", format='eps', dpi=1000)
    plt.close()

def save_plot_z_norm(mean_z_norm, model_used, z_dim, epochs, lr, batch_size):
    x = np.arange(1, len(mean_z_norm) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(x, mean_z_norm, label="Reconstruction error", linewidth=2)

    plt.axes().set_xlabel('Epoch')
    plt.axes().set_ylabel('Norm')
    plt.legend(loc='upper right')
    plt.suptitle("Evolution of the reconstruction error between z and E(G(z))")
    params = "Network type: " + model_used + ", Dimension of latent space: " + str(z_dim) + ", epochs: " + str(epochs) + ", learning rate: " + str(lr) + ", batch size:" + str(batch_size)
    plt.title(params, fontsize=8)
    # plt.show()
    plt.savefig("z_norm.eps", format='eps', dpi=1000)
    plt.close()

import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
import numpy as np
from .model import load_ckp
from scipy.stats import norm


def loss_fn(image_z1, image_z2, image_x_theta1, image_x_theta2, phi1, phi2, dim, w, scale):
    n = image_x_theta1.size(0)
    recon_loss1 = F.mse_loss(image_z1, image_x_theta1, reduction="sum").div(n)
    recon_loss2 = F.mse_loss(image_z2, image_x_theta2, reduction="sum").div(n)
    branch_loss = F.mse_loss(image_x_theta1, image_x_theta2, reduction="sum").div(n)
    if scale is True:
        trans_dim = 4
    else:
        trans_dim = 3
    z1_mean = phi1[:, trans_dim : trans_dim + dim]
    z1_var = phi1[:, -dim:]
    z2_mean = phi2[:, trans_dim : trans_dim + dim]
    z2_var = phi2[:, -dim:]
    dist_z1 = torch.distributions.multivariate_normal.MultivariateNormal(z1_mean, torch.diag_embed(z1_var.exp()))
    dist_z2 = torch.distributions.multivariate_normal.MultivariateNormal(z2_mean, torch.diag_embed(z2_var.exp()))
    z_loss = torch.mean(torch.distributions.kl.kl_divergence(dist_z1, dist_z2)).div(dim)
    loss = w * (recon_loss1 + recon_loss2 + branch_loss) + z_loss
    return loss


def plot_loss(epoch_train_loss, epoch_valid_loss):
    fig, ax = plt.subplots(dpi=150)
    train_loss_list = [x for x in epoch_train_loss]
    valid_loss_list = [x for x in epoch_valid_loss]
    (line1,) = ax.plot([i for i in range(len(train_loss_list))], train_loss_list)
    (line2,) = ax.plot([i for i in range(len(valid_loss_list))], valid_loss_list)
    ax.set_xlabel("Number of epochs")
    ax.set_ylabel("Loss")
    ax.legend((line1, line2), ("Train Loss", "Validation Loss"))
    plt.savefig("./harmony_results/Harmony_mnist_loss_curves.png", bbox_inches="tight")


def _save_sample_images(dataset_name, batch_size, recon_image, image, pixel):
    sample_out = recon_image.reshape(batch_size, pixel, pixel)
    plt.clf()
    fig = plt.figure(figsize=(8, 8))  # Notice the equal aspect ratio
    plot_per_row = math.ceil(batch_size**0.5)
    plot_per_col = math.ceil(batch_size / plot_per_row)
    ax = [fig.add_subplot(plot_per_row, plot_per_col, i + 1) for i in range(batch_size)]

    i = 0
    for a in ax:
        a.xaxis.set_visible(False)
        a.yaxis.set_visible(False)
        a.set_aspect("equal")
        a.imshow(sample_out[i], cmap="binary")
        i += 1

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("./harmony_results/Harmony_decoded_image_sample_" + dataset_name + ".png", bbox_inches="tight")

    sample_in = image.reshape(batch_size, pixel, pixel)
    plt.clf()
    fig = plt.figure(figsize=(8, 8))  # Notice the equal aspect ratio
    plot_per_row = round(batch_size**0.5)
    plot_per_col = round(batch_size / plot_per_row)
    ax = [fig.add_subplot(plot_per_row, plot_per_col, i + 1) for i in range(batch_size)]

    i = 0
    for a in ax:
        a.xaxis.set_visible(False)
        a.yaxis.set_visible(False)
        a.set_aspect("equal")
        a.imshow(sample_in[i], cmap="binary")
        i += 1

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("./harmony_results/Harmony_input_image_sample" + dataset_name + ".png", bbox_inches="tight")


def generate_manifold_images(dataset_name, trained_vae, pixel, z_dim=1, batch_size=100, device="cuda", n_per_dim=10):
    """
    Generate and save a grid of decoded images from a grid of latent vectors.
    For z_dim > 2, only the first two dimensions are varied in a grid, others are set to zero.
    Args:
        dataset_name (str): Name for saving.
        trained_vae: Trained VAE model.
        pixel (int): Image size.
        z_dim (int): Latent dimension.
        batch_size (int): Number of images to generate (ignored if z_dim > 2).
        device (str): Device.
        n_per_dim (int): Number of grid points per dimension for z_dim > 2.
    """
    import math

    trained_vae.eval()
    decoder = trained_vae.autoencoder.decoder

    if z_dim == 1:
        z_arr = norm.ppf(np.linspace(0.05, 0.95, batch_size))
        z = torch.from_numpy(z_arr).float().to(device=device)
        z = torch.unsqueeze(z, 1)
        image_z = decoder(z)
        manifold = image_z.cpu().detach()
        sample_out = manifold.reshape(batch_size, pixel, pixel)
        plt.clf()
        fig = plt.figure(figsize=(8, 8))
        plot_per_row = math.ceil(batch_size**0.5)
        plot_per_col = math.ceil(batch_size / plot_per_row)
        ax = [fig.add_subplot(plot_per_row, plot_per_col, i + 1) for i in range(batch_size)]
        for i, a in enumerate(ax):
            a.xaxis.set_visible(False)
            a.yaxis.set_visible(False)
            a.set_aspect("equal")
            a.imshow(sample_out[i], cmap="binary")
        fig.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(f"Harmony_manifold_image_{dataset_name}.png", bbox_inches="tight")
    else:
        # For z_dim >= 2, only vary the first two dimensions, others set to zero
        grid_x = norm.ppf(np.linspace(0.05, 0.95, n_per_dim))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, n_per_dim))
        z_list = []
        for yi in grid_x:
            for xi in grid_y:
                z_sample = np.zeros(z_dim)
                z_sample[0] = xi
                z_sample[1] = yi
                z_list.append(z_sample)
        z_arr = np.array(z_list)
        z = torch.from_numpy(z_arr).float().to(device=device)
        image_z = decoder(z)
        manifold = image_z.cpu().detach()
        sample_out = manifold.reshape(n_per_dim * n_per_dim, pixel, pixel)
        plt.clf()
        fig = plt.figure(figsize=(8, 8))
        ax = [fig.add_subplot(n_per_dim, n_per_dim, i + 1) for i in range(n_per_dim * n_per_dim)]
        for i, a in enumerate(ax):
            a.xaxis.set_visible(False)
            a.yaxis.set_visible(False)
            a.set_aspect("equal")
            a.imshow(sample_out[i], cmap="binary")
        fig.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(f"Harmony_manifold_image_{dataset_name}_zdim{z_dim}.png", bbox_inches="tight")


def save_output_images(dataset_name, test_loader, trained_model, pixel, type="test", batch_size=100, device="cuda"):
    trained_model.eval()
    all_images = []
    for batch_idx, images in enumerate(test_loader):
        with torch.no_grad():
            images = images.to(device=device)
            data = images.reshape(batch_size, 1, pixel, pixel)
            image_z1, image_z2, image_x_theta1, image_x_theta2, phi1, phi2 = trained_model(data)
            pose_image = image_z1.cpu().detach().numpy()
            all_images.append(pose_image)
    output_image = np.array(all_images)
    f = open("Harmony_decoded_images_" + dataset_name + "_" + type + ".pkl", "wb")
    pickle.dump(output_image, f)
    f.close()


def plot_sample_images(dataset_name, test_loader, trained_model, pixel, batch_size=100, device="cuda"):
    trained_model.eval()
    for batch_idx, images in enumerate(test_loader):
        with torch.no_grad():
            images = images.to(device=device)
            data = images.reshape(batch_size, 1, pixel, pixel)
            image_z1, image_z2, image_x_theta1, image_x_theta2, phi1, phi2 = trained_model(data)
            break
    pose_image = image_z1.cpu().detach()
    input_image = data.cpu().detach()
    _save_sample_images(dataset_name, batch_size, pose_image, input_image, pixel)


def save_latent_variables(dataset_name, data_loader, siamese, type, pixel, scale, batch_size=100, device="cuda"):
    Allphi = []
    siamese.eval()
    count = 0
    for batch_idx, images in enumerate(data_loader):
        count += 1
        with torch.no_grad():
            images = images.to(device=device)
            data = images.reshape(batch_size, 1, pixel, pixel)
            image_z1, image_z2, image_x_theta1, image_x_theta2, phi1, phi2 = siamese(data)
            phi_np = phi1.cpu().detach().numpy()
            Allphi.append(phi_np)
    if scale:
        z_dim = (phi_np.shape[1] - 4) // 2
    else:
        z_dim = (phi_np.shape[1] - 3) // 2
    PhiArr = np.array(Allphi).reshape(count * batch_size, -1)
    filepath = "Harmony_latent_factors_" + dataset_name + "_" + type + "z_dim_" + str(z_dim) + ".np"
    np.savetxt(filepath, PhiArr)

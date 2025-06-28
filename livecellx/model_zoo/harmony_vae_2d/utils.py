import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
from .model import load_ckp
from scipy.stats import norm


def loss_fn(
    image_z1, image_z2, image_x_theta1, image_x_theta2, phi1, phi2, dim, w, scale, return_dict=False, loss_z_factor=1.0
):
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
    loss = w * (recon_loss1 + recon_loss2 + branch_loss) + z_loss * loss_z_factor
    if return_dict:
        return {
            "recon_loss1": recon_loss1,
            "recon_loss2": recon_loss2,
            "branch_loss": branch_loss,
            "z_loss": z_loss,
            "total_loss": loss,
        }
    return loss


def plot_loss(
    dataset_name,
    epoch_train_loss,
    epoch_valid_loss,
    train_recon_loss1=None,
    train_recon_loss2=None,
    train_branch_loss=None,
    train_z_loss=None,
    val_recon_loss1=None,
    val_recon_loss2=None,
    val_branch_loss=None,
    val_z_loss=None,
    train_z_loss_weighted=None,
    val_z_loss_weighted=None,
    train_total_recon_weighted=None,
    val_total_recon_weighted=None,
):
    """
    Visualize all loss types (total, recon1, recon2, branch, z, weighted z) for train/val in one figure.
    Also plot:
      - Only total weighted reconstruction loss and weighted z loss
      - All reconstruction losses (recon1, recon2) in one figure (weighted)
      - All reconstruction losses (recon1, recon2) in one figure (unweighted)
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
    epochs = list(range(len(epoch_train_loss)))
    # Plot total loss
    ax.plot(epochs, epoch_train_loss, label="Train Total Loss", color="tab:blue")
    ax.plot(epochs, epoch_valid_loss, label="Val Total Loss", color="tab:orange")
    # Optionally plot sub-losses if provided
    if train_recon_loss1 is not None and val_recon_loss1 is not None:
        ax.plot(epochs, train_recon_loss1, label="Train Recon1", linestyle="--", color="tab:green")
        ax.plot(epochs, val_recon_loss1, label="Val Recon1", linestyle="--", color="tab:green", alpha=0.5)
    if train_recon_loss2 is not None and val_recon_loss2 is not None:
        ax.plot(epochs, train_recon_loss2, label="Train Recon2", linestyle="--", color="tab:red")
        ax.plot(epochs, val_recon_loss2, label="Val Recon2", linestyle="--", color="tab:red", alpha=0.5)
    if train_branch_loss is not None and val_branch_loss is not None:
        ax.plot(epochs, train_branch_loss, label="Train Branch", linestyle="--", color="tab:purple")
        ax.plot(epochs, val_branch_loss, label="Val Branch", linestyle="--", color="tab:purple", alpha=0.5)
    if train_z_loss is not None and val_z_loss is not None:
        ax.plot(epochs, train_z_loss, label="Train Z (KL)", linestyle="--", color="tab:brown")
        ax.plot(epochs, val_z_loss, label="Val Z (KL)", linestyle="--", color="tab:brown", alpha=0.5)
    if train_z_loss_weighted is not None and val_z_loss_weighted is not None:
        ax.plot(epochs, train_z_loss_weighted, label="Train Z (KL) Weighted", linestyle=":", color="tab:gray")
        ax.plot(epochs, val_z_loss_weighted, label="Val Z (KL) Weighted", linestyle=":", color="tab:gray", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves: Total and Components")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_dir = f"./harmony_results/{dataset_name}"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "Harmony_all_loss_curves.png"), bbox_inches="tight")
    plt.close()

    # Plot all reconstruction losses (weighted, NO branch, NO weighted z) in one figure
    if (
        train_recon_loss1 is not None
        and train_recon_loss2 is not None
        and val_recon_loss1 is not None
        and val_recon_loss2 is not None
    ):
        fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
        ax3.plot(epochs, train_recon_loss1, label="Train Recon1 (weighted)", color="tab:green")
        ax3.plot(epochs, train_recon_loss2, label="Train Recon2 (weighted)", color="tab:red")
        ax3.plot(epochs, val_recon_loss1, label="Val Recon1 (weighted)", linestyle="--", color="tab:green", alpha=0.5)
        ax3.plot(epochs, val_recon_loss2, label="Val Recon2 (weighted)", linestyle="--", color="tab:red", alpha=0.5)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Weighted Recon Loss")
        ax3.set_title("Recon1 & Recon2 Losses (Weighted)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, "Harmony_recon1_recon2_weighted_loss_curves.png"), bbox_inches="tight")
        plt.close()

    # Plot all reconstruction losses (unweighted, NO branch, NO weighted z) in one figure
    if (
        train_recon_loss1 is not None
        and train_recon_loss2 is not None
        and val_recon_loss1 is not None
        and val_recon_loss2 is not None
    ):
        fig4, ax4 = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
        ax4.plot(epochs, train_recon_loss1, label="Train Recon1", color="tab:green")
        ax4.plot(epochs, train_recon_loss2, label="Train Recon2", color="tab:red")
        ax4.plot(epochs, val_recon_loss1, label="Val Recon1", linestyle="--", color="tab:green", alpha=0.5)
        ax4.plot(epochs, val_recon_loss2, label="Val Recon2", linestyle="--", color="tab:red", alpha=0.5)
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Recon Loss (unweighted)")
        ax4.set_title("Recon1 & Recon2 Losses (Unweighted)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir, "Harmony_recon1_recon2_loss_curves.png"), bbox_inches="tight")
        plt.close()

    # Plot only weighted z loss (train/val)
    if train_z_loss_weighted is not None and val_z_loss_weighted is not None:
        fig5, ax5 = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
        ax5.plot(epochs, train_z_loss_weighted, label="Train Z (KL) Weighted", color="tab:gray")
        ax5.plot(epochs, val_z_loss_weighted, label="Val Z (KL) Weighted", linestyle="--", color="tab:gray", alpha=0.7)
        ax5.set_xlabel("Epoch")
        ax5.set_ylabel("Weighted Z Loss")
        ax5.set_title("Weighted Z (KL) Loss Only")
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        out_dir = f"./harmony_results/{dataset_name}"
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, "Harmony_weighted_z_loss_only_curves.png"), bbox_inches="tight")
        plt.close()


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
    plt.savefig(
        "./harmony_results/{dataset_name}/Harmony_decoded_image_sample_" + dataset_name + ".png", bbox_inches="tight"
    )

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
    plt.savefig(
        "../harmony_results/{dataset_name}/Harmony_input_image_sample" + dataset_name + ".png", bbox_inches="tight"
    )


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

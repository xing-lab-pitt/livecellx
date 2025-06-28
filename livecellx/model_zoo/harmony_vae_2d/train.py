import torch
from tqdm import tqdm
from .utils import loss_fn, plot_loss
from .model import save_ckp, load_ckp


def train_model(
    dataset_name,
    siamese,
    optimizer,
    train_loader,
    test_loader,
    device,
    start_epoch,
    n_epochs,
    epoch_train_loss,
    epoch_valid_loss,
    valid_loss_min,
    z_dim=2,
    pixel=64,
    batch_size=100,
    w=1,
    scale=False,
    loss_z_factor=1.0,
):
    print("[INFO] Training model on dataset: {} with parameters below".format(dataset_name))
    print("[INFO] Using z_dim: {}".format(z_dim))
    print("[INFO] Using pixel: {}".format(pixel))
    print("[INFO] Using batch_size: {}".format(batch_size))
    print("[INFO] Using w: {}".format(w))
    print("[INFO] Using scale: {}".format(scale))

    for epoch in tqdm(range(start_epoch, n_epochs + 1), desc="Training Epochs"):
        train_loss = 0.0
        valid_loss = 0.0
        siamese.train()
        train_recon_loss1 = 0.0
        train_recon_loss2 = 0.0
        train_branch_loss = 0.0
        train_z_loss = 0.0
        for batch_idx, images in tqdm(enumerate(train_loader), desc="Training Batches", total=len(train_loader)):
            images = images.to(device=device)
            data = images.reshape(batch_size, 1, pixel, pixel)
            image_z1, image_z2, image_x_theta1, image_x_theta2, phi1, phi2 = siamese(data, scale)
            loss_dict = loss_fn(
                image_z1, image_z2, image_x_theta1, image_x_theta2, phi1, phi2, z_dim, w, scale, return_dict=True
            )
            loss = loss_dict["total_loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_recon_loss1 += loss_dict["recon_loss1"].item()
            train_recon_loss2 += loss_dict["recon_loss2"].item()
            train_branch_loss += loss_dict["branch_loss"].item()
            train_z_loss += loss_dict["z_loss"].item()
        epoch_train_loss.append(train_loss / len(train_loader))
        avg_train_recon_loss1 = train_recon_loss1 / len(train_loader)
        avg_train_recon_loss2 = train_recon_loss2 / len(train_loader)
        avg_train_branch_loss = train_branch_loss / len(train_loader)
        avg_train_z_loss = train_z_loss / len(train_loader)
        # validate the model #
        ######################
        siamese.eval()
        val_recon_loss1 = 0.0
        val_recon_loss2 = 0.0
        val_branch_loss = 0.0
        val_z_loss = 0.0
        for batch_idx, images in enumerate(test_loader):
            with torch.no_grad():
                images = images.to(device=device)
                data = images.reshape(batch_size, 1, pixel, pixel)
                image_z1, image_z2, image_x_theta1, image_x_theta2, phi1, phi2 = siamese(data, scale)
                val_loss_dict = loss_fn(
                    image_z1,
                    image_z2,
                    image_x_theta1,
                    image_x_theta2,
                    phi1,
                    phi2,
                    z_dim,
                    w,
                    scale,
                    return_dict=True,
                    loss_z_factor=loss_z_factor,
                )
                valid_loss += val_loss_dict["total_loss"].item()
                val_recon_loss1 += val_loss_dict["recon_loss1"].item()
                val_recon_loss2 += val_loss_dict["recon_loss2"].item()
                val_branch_loss += val_loss_dict["branch_loss"].item()
                val_z_loss += val_loss_dict["z_loss"].item()
        epoch_valid_loss.append(valid_loss / len(test_loader))
        avg_val_recon_loss1 = val_recon_loss1 / len(test_loader)
        avg_val_recon_loss2 = val_recon_loss2 / len(test_loader)
        avg_val_branch_loss = val_branch_loss / len(test_loader)
        avg_val_z_loss = val_z_loss / len(test_loader)
        # Store per-epoch loss values for plotting
        if epoch == start_epoch:
            train_recon_loss1_curve = []
            train_recon_loss2_curve = []
            train_branch_loss_curve = []
            train_z_loss_curve = []
            val_recon_loss1_curve = []
            val_recon_loss2_curve = []
            val_branch_loss_curve = []
            val_z_loss_curve = []
        train_recon_loss1_curve.append(avg_train_recon_loss1)
        train_recon_loss2_curve.append(avg_train_recon_loss2)
        train_branch_loss_curve.append(avg_train_branch_loss)
        train_z_loss_curve.append(avg_train_z_loss)
        val_recon_loss1_curve.append(avg_val_recon_loss1)
        val_recon_loss2_curve.append(avg_val_recon_loss2)
        val_branch_loss_curve.append(avg_val_branch_loss)
        val_z_loss_curve.append(avg_val_z_loss)
        # print training/validation statistics
        print(
            f"Epoch: {epoch}\tTraining Loss: {epoch_train_loss[epoch]:.6f}\tValidation Loss: {epoch_valid_loss[epoch]:.6f}\n"
            f"  Train Recon1: {avg_train_recon_loss1:.6f}  Train Recon2: {avg_train_recon_loss2:.6f}  Train Branch: {avg_train_branch_loss:.6f}  Train Z: {avg_train_z_loss:.6f} Weighted Z Loss: {loss_z_factor * avg_val_z_loss:.6f}\n"
            f"  Val Recon1: {avg_val_recon_loss1:.6f}  Val Recon2: {avg_val_recon_loss2:.6f}  Val Branch: {avg_val_branch_loss:.6f}  Val Z: {avg_val_z_loss:.6f} Weighted Z Loss: {loss_z_factor * avg_val_z_loss:.6f}\n",
        )

        if epoch % 10 == 0 and epoch_valid_loss[epoch] < valid_loss_min:
            print(
                "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
                    valid_loss_min, epoch_valid_loss[epoch]
                )
            )
            # save checkpoint as best model

            checkpoint = {
                "epoch": epoch + 1,
                "valid_loss_min": epoch_valid_loss[epoch],
                "epoch_train_loss": epoch_train_loss,
                "epoch_valid_loss": epoch_valid_loss,
                "state_dict": siamese.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_ckp(
                checkpoint,
                f"./harmony_results/{dataset_name}/best_model_Harmony_fc"
                + dataset_name
                + "_z_dim_{}_w_{}.pt".format(z_dim, w),
            )
            valid_loss_min = epoch_valid_loss[epoch]
        # Compute total weighted reconstruction loss for each epoch
        train_total_recon_weighted = [
            w * (r1 + r2 + b)
            for r1, r2, b in zip(train_recon_loss1_curve, train_recon_loss2_curve, train_branch_loss_curve)
        ]
        val_total_recon_weighted = [
            w * (r1 + r2 + b) for r1, r2, b in zip(val_recon_loss1_curve, val_recon_loss2_curve, val_branch_loss_curve)
        ]
        plot_loss(
            dataset_name,
            epoch_train_loss=epoch_train_loss,
            epoch_valid_loss=epoch_valid_loss,
            train_recon_loss1=train_recon_loss1_curve,
            train_recon_loss2=train_recon_loss2_curve,
            train_branch_loss=train_branch_loss_curve,
            train_z_loss=train_z_loss_curve,
            val_recon_loss1=val_recon_loss1_curve,
            val_recon_loss2=val_recon_loss2_curve,
            val_branch_loss=val_branch_loss_curve,
            val_z_loss=val_z_loss_curve,
            train_z_loss_weighted=[z * loss_z_factor for z in train_z_loss_curve],
            val_z_loss_weighted=[z * loss_z_factor for z in val_z_loss_curve],
            train_total_recon_weighted=train_total_recon_weighted,
            val_total_recon_weighted=val_total_recon_weighted,
        )

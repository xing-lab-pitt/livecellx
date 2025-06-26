import torch
import numpy as np
from livecellx.model_zoo.harmony_vae_2d.model import get_instance_model_optimizer, load_ckp
from livecellx.model_zoo.harmony_vae_2d.data import data_loader, estimate_optimal_gamma
from livecellx.model_zoo.harmony_vae_2d.train import train_model
from livecellx.model_zoo.harmony_vae_2d.evaluate import evaluate_model
from livecellx.model_zoo.harmony_vae_2d.sc_dataloader import scs_train_test_dataloader
import argparse


def train_and_evaluate(
    dataset_name,
    batch_size=100,
    n_epochs=5,
    learning_rate=0.0001,
    z_dim=2,
    pixel=64,
    load_model=False,
    w=1,
    scale=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    siamese, optimizer = get_instance_model_optimizer(device, learning_rate, z_dim, pixel)
    if dataset_name == "MCF10A_and_A549":
        print("[INFO] Using MCF10A_and_A549 dataset")
        train_loader, test_loader = scs_train_test_dataloader(
            padding=30, img_shape=(pixel, pixel), batch_size=batch_size
        )
    else:
        train_loader, test_loader, mu, std = data_loader(dataset_name, pixel, batch_size)
    print("[INFO] Using dataset: {}".format(dataset_name))
    print("[INFO] Using z_dim: {}".format(z_dim))
    print("[INFO] Using pixel: {}".format(pixel))
    print("[INFO] Using batch_size: {}".format(batch_size))
    print("[INFO] Using learning_rate: {}".format(learning_rate))
    print("[INFO] Using w: {}".format(w))
    # Print dataset size
    print("[INFO] Training dataset size: {}".format(len(train_loader.dataset)))
    print("[INFO] Test dataset size: {}".format(len(test_loader.dataset)))

    if load_model:
        (siamese, optimizer, start_epoch, epoch_train_loss, epoch_valid_loss, valid_loss_min,) = load_ckp(
            siamese,
            optimizer,
            "best_model_Harmony_fc" + dataset_name + "_z_dim_{}_w_{}.pt".format(z_dim, w),
        )
    else:
        valid_loss_min = np.inf
        start_epoch = 0
        epoch_train_loss = []
        epoch_valid_loss = []

    train_model(
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
        z_dim,
        pixel,
        batch_size,
        w,
        scale,
    )

    evaluate_model(dataset_name, siamese, z_dim, pixel, batch_size, device, scale)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train and Evaluate Harmony on your dataset")
    parser.add_argument("-z", "--z-dim", type=int, default=1)
    parser.add_argument("-bs", "--batch-size", type=int, default=100)
    parser.add_argument("-ep", "--num-epochs", type=int, default=20)
    parser.add_argument("-l", "--learning-rate", type=float, default=0.0001)
    parser.add_argument("--load-model", action="store_true")
    parser.add_argument("-w", "--gamma", type=int)
    parser.add_argument("--scale", action="store_true", default=False)
    parser.add_argument("-dat", "--dataset", type=str)
    parser.add_argument("-p", "--pixel", type=int, required=False)
    args = parser.parse_args()
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    z_dim = args.z_dim
    dataset_name = args.dataset
    if dataset_name == "MNIST":
        pixel = 28
    else:
        pixel = 40

    if args.pixel:  # If argument is provided it will supercede
        pixel = args.pixel
    load_model = False
    scale = False
    if args.load_model:
        load_model = True
    if args.scale:
        scale = True

    # dataset_name = "codhacs"

    if args.gamma:
        w = args.gamma
    else:
        if dataset_name == "MCF10A_and_A549":
            w = 166531 / (args.batch_size * 1000)  # For MCF10A_and_A549 dataset, use a fixed value
        else:
            w = estimate_optimal_gamma(dataset_name, batch_size)

    train_and_evaluate(
        dataset_name=dataset_name,
        batch_size=batch_size,
        n_epochs=num_epochs,
        learning_rate=learning_rate,
        z_dim=z_dim,
        pixel=pixel,
        load_model=load_model,
        w=w,
    )

import torch
import numpy as np
from livecellx.core.io_sc import prep_scs_from_mask_dataset
from livecellx.model_zoo.harmony_vae_2d.model import get_instance_model_optimizer, load_ckp
from livecellx.model_zoo.harmony_vae_2d.data import data_loader, estimate_optimal_gamma
from livecellx.model_zoo.harmony_vae_2d.train import train_model
from livecellx.model_zoo.harmony_vae_2d.evaluate import evaluate_model
from livecellx.model_zoo.harmony_vae_2d.sc_dataloader import scs_train_test_dataloader
import livecellx.model_zoo.harmony_vae_2d.harmony_model_conv
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
    *,
    decoder_type="fc",  # Added decoder_type argument
    debug=False,
    loss_z_factor=1.0,  # Added loss_z_factor argument
    include_background=True,  # New argument
    loss_version="v1",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if decoder_type == "v2_conv":
        print("[INFO] Using v2_conv decoder")
        (
            siamese,
            optimizer,
            scheduler,
        ) = livecellx.model_zoo.harmony_vae_2d.harmony_model_conv.get_instance_model_optimizer(
            device, learning_rate, z_dim, pixel
        )
    else:
        siamese, optimizer = get_instance_model_optimizer(
            device, learning_rate, z_dim, pixel, decoder_type=decoder_type
        )
    if debug:
        from livecellx.sample_data import tutorial_three_image_sys

        batch_size = 2
        dic_dataset, mask_dataset = tutorial_three_image_sys()
        single_cells = prep_scs_from_mask_dataset(mask_dataset, dic_dataset)
        train_loader, test_loader = scs_train_test_dataloader(
            scs=single_cells,
            padding=30,
            img_shape=(pixel, pixel),
            batch_size=batch_size,
            include_background=include_background,
        )

    elif dataset_name.startswith("MCF10A"):  # "MCF10A_and_A549":
        print("[INFO] Using MCF10A_and_A549 dataset")
        train_loader, test_loader = scs_train_test_dataloader(
            padding=30, img_shape=(pixel, pixel), batch_size=batch_size, include_background=include_background
        )
    else:
        train_loader, test_loader, mu, std = data_loader(dataset_name, pixel, batch_size)
    print("[INFO] Using dataset: {}".format(dataset_name))
    print("[INFO] Using z_dim: {}".format(z_dim))
    print("[INFO] Using pixel: {}".format(pixel))
    print("[INFO] Using batch_size: {}".format(batch_size))
    print("[INFO] Using learning_rate: {}".format(learning_rate))
    print("[INFO] Using w: {}".format(w))
    print("[INFO] Using scale: {}".format(scale))
    print("[INFO] Using decoder_type: {}".format(decoder_type))
    print("[INFO] Using loss_z_factor: {}".format(loss_z_factor))
    print("[INFO] Using include_background: {}".format(include_background))
    if scale:
        assert NotImplementedError(
            "Scaling is not implemented in this version. Please set scale=False or implement scaling in the model."
        )

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
        loss_z_factor=loss_z_factor,  # Pass the loss_z_factor argument
        loss_version=loss_version,  # Pass the loss_version argument
    )

    evaluate_model(dataset_name, siamese, z_dim, pixel, batch_size, device, scale, test_loader=test_loader)


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
    parser.add_argument("--decoder-type", type=str, default="fc", choices=["fc", "conv", "v2_conv"])
    parser.add_argument("--loss_z_factor", type=float, default=1.0, help="Weight for the z factor loss term")
    parser.add_argument(
        "--include_background",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Include background pixels in training images (default: True/False)",
    )
    parser.add_argument(
        "--loss_version", type=str, default="v1", choices=["v1", "v2"], help="Version of the loss function to use"
    )
    # debug flag
    parser.add_argument("--debug", action="store_true", default=False, help="Run in debug mode with sample data")
    args = parser.parse_args()
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    z_dim = args.z_dim
    dataset_name = args.dataset

    if args.pixel:  # If argument is provided it will supercede
        pixel = args.pixel
    elif dataset_name == "MNIST":
        print("[INFO] Using MNIST pixel size of 28")
        pixel = 28
    else:
        print("[INFO] No arg set, Using default pixel size of 64")
        pixel = 40

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
        if dataset_name.startswith("MCF10A") or dataset_name.startswith("tutorial"):  # == "MCF10A_and_A549":
            w = 166531 / (args.batch_size * 1000)  # For MCF10A_and_A549 dataset, use a fixed value
        elif dataset_name == "codhacs":
            w = estimate_optimal_gamma(dataset_name, batch_size)
        else:
            assert False, "Please provide a valid dataset name or gamma value"

    # Create ./harmony_results/{dataset_name}/
    from pathlib import Path

    Path(f"./harmony_results/{dataset_name}").mkdir(parents=True, exist_ok=True)
    train_and_evaluate(
        dataset_name=dataset_name,
        batch_size=batch_size,
        n_epochs=num_epochs,
        learning_rate=learning_rate,
        z_dim=z_dim,
        pixel=pixel,
        load_model=load_model,
        w=w,
        decoder_type=args.decoder_type,
        debug=args.debug,
        loss_z_factor=args.loss_z_factor,  # Pass the loss_z_factor argument
        include_background=args.include_background,  # Pass the include_background argument
        loss_version=args.loss_version,  # Pass the loss_version argument
    )

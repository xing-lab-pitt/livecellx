from livecell_tracker.core.torch_datasets import SingleCellVaeDataset, ScVaeDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from typing import List, Union
import torch
import os
import torchvision
from livecell_tracker.core import SingleCellStatic
from pathlib import Path
from skimage.measure import regionprops
from livecell_tracker.core.datasets import LiveCellImageDataset, SingleImageDataset
from livecell_tracker.preprocess.utils import normalize_img_to_uint8

from livecell_tracker.segment.utils import prep_scs_from_mask_dataset
from livecell_tracker.model_zoo.autoencoder.autoencoder import Autoencoder

dataset_dir_path = Path("../datasets/test_data_STAV-A549/DIC_data")
mask_dataset_path = Path("../datasets/test_data_STAV-A549/mask_data")
mask_dataset = LiveCellImageDataset(mask_dataset_path, ext="png")
dic_dataset = LiveCellImageDataset(dataset_dir_path, ext="tif")

################### large dataset ###################
# dataset_dir_path = Path("../datasets/EBSS_Starvation/tif_STAV-A549_VIM_24hours_NoTreat_NA_YL_Ti2e_2022-12-21/XY16/")

# mask_dataset_path = Path(
#     "../datasets/EBSS_Starvation/tif_STAV-A549_VIM_24hours_NoTreat_NA_YL_Ti2e_2022-12-21/out/XY16/seg"
# )
# mask_dataset = LiveCellImageDataset(mask_dataset_path, ext="png")
# import glob

# time2url = sorted(glob.glob(str((Path(dataset_dir_path) / Path("*_DIC.tif")))))
# time2url = {i: path for i, path in enumerate(time2url)}
# dic_dataset = LiveCellImageDataset(time2url=time2url, ext="tif")

single_cells = prep_scs_from_mask_dataset(mask_dataset, dic_dataset)
for sc in single_cells:
    sc.cache = False
config = {
    "model_params": {"name": "VanillaVAE", "in_channels": 1, "latent_dim": 128},
    "data_params": {
        "data_path": "Data/",
        "train_batch_size": 64,
        "val_batch_size": 64,
        "patch_size": 128,
        "num_workers": 0,
    },
}


class GenerateCallback(Callback):
    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs  # Images to reconstruct during training
        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(input_imgs)
                pl_module.train()
            # Plot and add to tensorboard
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1, 1))
            trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)


def train_scs(latent_dim, data: ScVaeDataset):
    # Create a PyTorch Lightning trainer with the generation callback
    CHECKPOINT_PATH = "./autoencoder_ckpts"
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "scs_%i" % latent_dim),
        accelerator="auto",
        devices=1,
        max_epochs=500,
        callbacks=[
            ModelCheckpoint(save_weights_only=True),
            GenerateCallback(torch.stack([data.train_dataset[i][0] for i in range(10)], dim=0), every_n_epochs=10),
            LearningRateMonitor("epoch"),
        ],
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "scs_%i.ckpt" % latent_dim)
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = Autoencoder.load_from_checkpoint(pretrained_filename)
    else:
        model = Autoencoder(
            base_channel_size=64,
            latent_dim=latent_dim,
            num_input_channels=config["model_params"]["in_channels"],
            width=config["data_params"]["patch_size"],
            height=config["data_params"]["patch_size"],
        )
        trainer.fit(model, datamodule=data)
    # Test best model on validation and test set
    # val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    # test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    # result = {"test": test_result, "val": val_result}
    return model  # , result


data = ScVaeDataset(single_cells, single_cells, **config["data_params"], img_only=False)
data.setup()
train_scs(64, data)

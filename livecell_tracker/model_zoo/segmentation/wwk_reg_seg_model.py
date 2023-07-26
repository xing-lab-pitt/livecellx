import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class RegSegModel(pl.LightningModule):
    def __init__(self):
        super(RegSegModel, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)

        # Decoder
        self.up6 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv6 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.up9 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.conv11 = nn.Conv2d(64, 1, kernel_size=1)
        self.dropout = nn.Dropout2d(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Encoder
        conv1 = self.relu(self.conv1(x))
        pool1 = nn.MaxPool2d(kernel_size=2)(conv1)

        conv2 = self.relu(self.conv2(pool1))
        pool2 = nn.MaxPool2d(kernel_size=2)(conv2)

        conv3 = self.relu(self.conv3(pool2))
        pool3 = nn.MaxPool2d(kernel_size=2)(conv3)

        conv4 = self.relu(self.conv4(pool3))
        pool4 = nn.MaxPool2d(kernel_size=2)(conv4)

        conv5 = self.relu(self.conv5(pool4))

        # Decoder
        up6 = self.up6(conv5)
        concat6 = torch.cat([up6, conv4], dim=1)
        conv6 = self.relu(self.conv6(concat6))

        up7 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)(conv6)
        concat7 = torch.cat([up7, conv3], dim=1)
        conv7 = self.relu(self.conv7(concat7))

        up8 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)(conv7)
        concat8 = torch.cat([up8, conv2], dim=1)
        conv8 = self.relu(self.conv8(concat8))

        up9 = self.up9(conv8)
        concat9 = torch.cat([up9, conv1], dim=1)
        conv9 = self.relu(self.conv9(concat9))

        conv10 = self.relu(self.conv10(conv9))

        conv11 = self.conv11(conv10)
        drop11 = self.dropout(conv11)
        outputs = self.relu(drop11)

        return outputs

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        acc = (y_hat.round() == y).float().mean()
        iou = (y_hat.round() & y).float().sum() / (y_hat.round() | y).float().sum()
        self.log_dict({"val_loss": loss, "val_mae": mae, "val_acc": acc, "val_iou": iou})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

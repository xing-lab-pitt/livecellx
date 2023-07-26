import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class RegThreeStackConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(RegThreeStackConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        return x


class RegSegModel(pl.LightningModule):
    def __init__(self, padding=1):
        super(RegSegModel, self).__init__()

        # Encoder
        self.conv1 = RegThreeStackConvBlock(1, 64, kernel_size=3, padding=padding)
        self.conv2 = RegThreeStackConvBlock(64, 128, kernel_size=3, padding=padding)
        self.conv3 = RegThreeStackConvBlock(128, 256, kernel_size=3, padding=padding)
        self.conv4 = RegThreeStackConvBlock(256, 512, kernel_size=3, padding=padding)
        self.conv5 = RegThreeStackConvBlock(512, 1024, kernel_size=3, padding=padding)

        # Decoder
        self.up6 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv6 = RegThreeStackConvBlock(1024, 512, kernel_size=3, padding=padding)
        self.conv7 = RegThreeStackConvBlock(512, 256, kernel_size=3, padding=padding)
        self.conv8 = RegThreeStackConvBlock(256, 128, kernel_size=3, padding=padding)

        self.up9 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv9 = RegThreeStackConvBlock(128, 128, kernel_size=3, padding=padding)
        self.conv10 = RegThreeStackConvBlock(128, 64, kernel_size=3, padding=padding)

        self.conv11 = RegThreeStackConvBlock(64, 1, kernel_size=1, padding=0)
        self.dropout = nn.Dropout2d(p=0.5)
        self.relu = nn.ReLU()
        # debug: print conv1 to conv11 shape
        # print("conv1 shape:", self.conv1)
        # print("conv2 shape:", self.conv2)
        # print("conv3 shape:", self.conv3)
        # print("conv4 shape:", self.conv4)
        # print("conv5 shape:", self.conv5)
        # print("conv6 shape:", self.conv6)
        # print("conv7 shape:", self.conv7)
        # print("conv8 shape:", self.conv8)
        # print("conv9 shape:", self.conv9)
        # print("conv10 shape:", self.conv10)

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
        # concat6 = torch.cat([up6, conv4], dim=1)
        conv6 = self.relu(self.conv6(up6))

        up7 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)(conv6)
        # concat7 = torch.cat([up7, conv3], dim=1)
        conv7 = self.relu(self.conv7(up7))

        up8 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)(conv7)
        # concat8 = torch.cat([up8, conv2], dim=1)
        conv8 = self.relu(self.conv8(up8))

        up9 = self.up9(conv8)
        # concat9 = torch.cat([up9, conv1], dim=1)
        conv9 = self.relu(self.conv9(up9))

        conv10 = self.relu(self.conv10(conv9))

        conv11 = self.conv11(conv10)
        drop11 = self.dropout(conv11)
        outputs = self.relu(drop11)
        return outputs

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # print(f"shape of y_hat: {y_hat.shape}", f"shape of y: {y.shape}")
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        threshold = 0.5
        x, y = batch
        y_hat = self(x)
        mse_loss = F.mse_loss(y_hat, y)
        l1_loss = F.l1_loss(y_hat, y)
        acc = ((y_hat > threshold) == (y >= 1)).float().mean()

        # compute IOU
        y_hat = y_hat > threshold  # 0.5 as threshold
        y = y.int()

        intersection = (y_hat & (y >= 1)).sum()
        union = (y_hat | y).sum()

        if union > 0:
            iou = intersection / union
        else:
            iou = torch.tensor(0.0)

        self.log_dict({"val_loss": mse_loss, "val_l1_loss": l1_loss, "val_acc": acc, f"val_iou_{threshold}=0.5": iou})
        return mse_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)

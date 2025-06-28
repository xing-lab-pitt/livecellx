import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from livecellx.model_zoo.harmony_vae_2d.model import Transformer


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(0.25)

        # Match dimension
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        out += identity
        out = self.relu(out)
        return out


class ReverseResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super(ReverseResidualBlock, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size, 1, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(0.25)

        # match dimention
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.deconv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.deconv2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        out += identity
        out = self.relu(out)
        return out


class Encoder(nn.Module):
    def __init__(self, latent_dims, pixel):
        super(Encoder, self).__init__()
        self.res1 = ResidualBlock(1, 32, 3, 2, 1)
        self.res2 = ResidualBlock(32, 32, 3, 2, 1)
        self.res3 = ResidualBlock(32, 64, 3, 2, 1)
        self.res4 = ResidualBlock(64, 64, 3, 2, 1)
        self.res5 = ResidualBlock(64, 128, 5, 2, 2)
        self.res6 = ResidualBlock(128, 512, 5, 2, 2)
        self.conv7 = nn.Conv2d(512, 1 + 2 * latent_dims, 3, 2, 0)
        self.latent_dims = latent_dims

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.conv7(x)
        phi = x.view(x.size(0), 1 + 2 * self.latent_dims)
        return phi


class Decoder(nn.Module):
    def __init__(self, latent_dims, pixel):
        super(Decoder, self).__init__()
        self.res1 = ReverseResidualBlock(latent_dims, 512, 3, 2, 0, 1)
        self.res2 = ReverseResidualBlock(512, 128, 5, 2, 2, 0)
        self.res3 = ReverseResidualBlock(128, 64, 5, 2, 2, 0)
        self.res4 = ReverseResidualBlock(64, 64, 3, 2, 1, 0)
        self.res5 = ReverseResidualBlock(64, 32, 5, 2, 2, 1)
        self.res6 = ReverseResidualBlock(32, 32, 3, 2, 1, 1)
        self.rev_conv = nn.ConvTranspose2d(32, 1, 3, 2, 1, 1)
        self.activation = nn.ReLU()
        self.pixel = pixel

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        x = self.res1(z)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)

        x = self.rev_conv(x)

        x = self.activation(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, latent_dims, pixel):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(latent_dims, pixel)
        self.decoder = Decoder(latent_dims, pixel)
        self.transform = Transformer()
        self.latent_dims = latent_dims
        self.pixel = pixel

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, scale):
        phi = self.encoder(x)

        theta = phi[:, 0]
        # translations = phi[:, 1:3]
        translations = None

        if scale == False:
            image_x_theta = self.transform(x, theta, translations, None)
            z_mu = phi[:, 1 : 1 + self.latent_dims]
        else:
            if translations is None:
                scaling_factor = F.hardtanh(phi[:, 1], 0.8, 1.2)
                # scaling_factor = F.relu(phi[:, 1])
                z_mu = phi[:, 2 : 2 + self.latent_dims]
            else:
                scaling_factor = phi[:, 3]
                z_mu = phi[:, 4 : 4 + self.latent_dims]
            image_x_theta = self.transform(x, theta, translations, scaling_factor)
        z_var = phi[:, -self.latent_dims :]
        z = self.reparametrize(z_mu, z_var)

        image_z = self.decoder.forward(z)

        return image_z, image_x_theta, phi


class Siamese(nn.Module):
    def __init__(self, latent_dims, pixel):
        super(Siamese, self).__init__()
        self.autoencoder = AutoEncoder(latent_dims, pixel)
        self.transform = Transformer()
        self.pixel = pixel

    def forward(self, image, scale=False):
        with torch.no_grad():
            rot_theta = torch.FloatTensor(image.size(0), 1).uniform_(-np.pi / 2, np.pi / 2).to(device=image.device)
            translations = None  # torch.FloatTensor(image.size(0), 2).uniform_(-3, 3).to(device=image.device)
            if scale == True:
                scaling_factor = torch.FloatTensor(image.size(0), 1).uniform_(1.0, 1.5).to(device=image.device)
                transformed_image = self.transform(image, rot_theta, translations, scaling_factor)
            else:
                transformed_image = self.transform(image, rot_theta, translations, None)
        image_z1, image_x_theta1, phi1 = self.autoencoder(image, scale)
        image_z2, image_x_theta2, phi2 = self.autoencoder(transformed_image, scale)

        return image_z1, image_z2, image_x_theta1, image_x_theta2, phi1, phi2


def load_ckp(model, optimizer=None, f_path="./best_model.pt"):
    # load check point
    checkpoint = torch.load(f_path)

    model.load_state_dict(checkpoint["state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer"])

    valid_loss_min = checkpoint["valid_loss_min"]
    epoch_train_loss = checkpoint["epoch_train_loss"]
    epoch_valid_loss = checkpoint["epoch_valid_loss"]

    return model, optimizer, checkpoint["epoch"], epoch_train_loss, epoch_valid_loss, valid_loss_min


def save_ckp(state, f_path="./best_model.pt"):
    torch.save(state, f_path)


def get_instance_model_optimizer(device, learning_rate=0.0001, z_dims=2, pixel=64):
    print("Loading model")
    model = Siamese(latent_dims=z_dims, pixel=pixel).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=3)
    return model, optimizer, scheduler

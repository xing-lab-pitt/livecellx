import numpy as np

# import kornia  # version 0.4.0
import torch
import kornia
import kornia.geometry.transform as tf

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, latent_dims, pixel):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(pixel * pixel, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 128)
        self.fc6 = nn.Linear(128, 2 * latent_dims + 3)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.fc4(x))
        x = F.tanh(self.fc5(x))
        phi = self.fc6(x)
        return phi


class Transformer__kornia_deprecated(object):
    def __call__(self, image, theta, translations, scale_factor):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        B, C, H, W = image.size()
        center = torch.tensor([H / 2, W / 2]).repeat(B, 1).to(device=device)
        angle = theta.squeeze()
        angle = torch.rad2deg(angle)
        rotated_im = kornia.rotate(image, angle, center=center, padding_mode="zeros", align_corners=False)
        transformed_im = kornia.translate(rotated_im, translations, padding_mode="zeros", align_corners=False)
        if scale_factor is not None:
            transformed_im = kornia.scale(transformed_im, scale_factor, padding_mode="zeros", align_corners=False)
        return transformed_im


class Transformer(object):
    def __call__(self, image, theta, translations, scale_factor):
        device = image.device
        B, C, H, W = image.size()
        # Center as (x, y) = (W/2, H/2)
        center = torch.tensor([[W / 2, H / 2]], device=device).repeat(B, 1)
        # theta: shape (B, 1) or (B,), convert radians to degrees for Kornia
        angle = theta.view(-1)
        angle_deg = torch.rad2deg(angle)

        # Step 1: Rotation matrix
        # Ensure scale_factor is (B, 2)
        if scale_factor is None:
            scale = torch.ones(B, 2, device=device)
        elif scale_factor.dim() == 1:
            scale = scale_factor.unsqueeze(-1).repeat(1, 2)
        else:
            scale = scale_factor  # Already correct shape
        rot_mat = tf.get_rotation_matrix2d(center, angle_deg, scale)

        # Step 2: Add translation (to affine matrix, last column)
        if translations is not None:
            rot_mat = rot_mat.clone()
            rot_mat[:, :, 2] += translations

        # Step 3: Apply warp_affine for rotation+translation
        rotated_translated = tf.warp_affine(
            image, rot_mat, dsize=(H, W), mode="bilinear", padding_mode="zeros", align_corners=False
        )

        # Step 4: Optional scaling (scale around center)
        if scale_factor is not None:
            # Create new affine for scaling only (angle=0)
            scale_mat = tf.get_rotation_matrix2d(center, torch.zeros(B, device=device), scale_factor)
            out = tf.warp_affine(
                rotated_translated, scale_mat, dsize=(H, W), mode="bilinear", padding_mode="zeros", align_corners=False
            )
        else:
            out = rotated_translated

        return out


class Decoder(nn.Module):
    def __init__(self, latent_dims, pixel):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dims, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, pixel * pixel)
        self.pixel = pixel

    def forward(self, z):
        x = F.tanh(self.fc1(z))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.fc4(x))
        x = F.tanh(self.fc5(x))
        x = F.tanh(self.fc6(x))
        x = x.view(x.size(0), 1, self.pixel, self.pixel)
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
        translations = phi[:, 1:3]
        if scale == False:
            image_x_theta = self.transform(x, theta, translations, None)
            z_mu = phi[:, 3 : 3 + self.latent_dims]
        else:
            scaling_factor = phi[:, 3]
            image_x_theta = self.transform(x, theta, translations, scaling_factor)
            z_mu = phi[:, 4 : 4 + self.latent_dims]
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
            translations = torch.FloatTensor(image.size(0), 2).uniform_(-3, 3).to(device=image.device)
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
    return model, optimizer

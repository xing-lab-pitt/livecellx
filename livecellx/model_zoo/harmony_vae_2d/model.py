import numpy as np

# import kornia  # version 0.4.0
import torch
import kornia
import kornia.geometry.transform as tf

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.functional as F

# class LinearEncoder(nn.Module):
#     def __init__(self, latent_dims, pixel):
#         super(Encoder, self).__init__()
#         self.fc1 = nn.Linear(pixel * pixel, 512)
#         self.fc2 = nn.Linear(512, 512)
#         self.fc3 = nn.Linear(512, 512)
#         self.fc4 = nn.Linear(512, 512)
#         self.fc5 = nn.Linear(512, 128)
#         self.fc6 = nn.Linear(128, 2 * latent_dims + 3)

#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         x = F.tanh(self.fc1(x))
#         x = F.tanh(self.fc2(x))
#         x = F.tanh(self.fc3(x))
#         x = F.tanh(self.fc4(x))
#         x = F.tanh(self.fc5(x))
#         phi = self.fc6(x)
#         return phi


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
    def __call__(self, image, theta, translations=None, scale_factor=None):
        device = image.device
        B, C, H, W = image.size()

        # Rotation (around image center)
        center = torch.tensor([[W / 2, H / 2]], device=device).repeat(B, 1)
        angle_deg = torch.rad2deg(theta.view(-1))

        scale_rot = torch.ones(B, 2, device=device)  # No scaling during rotation
        rot_mat = tf.get_rotation_matrix2d(center, angle_deg, scale_rot)  # shape (B, 2, 3)

        rotated = tf.warp_affine(
            image, rot_mat, dsize=(H, W), mode="bilinear", padding_mode="reflection", align_corners=False
        )

        # Translation (after rotation)
        out = rotated
        if translations is not None:
            # Build translation affine: identity rotation, translation in pixels
            translation_mat = torch.eye(2, 3, device=device).unsqueeze(0).repeat(B, 1, 1)
            translation_mat[:, 0, 2] = translations[:, 0]
            translation_mat[:, 1, 2] = translations[:, 1]
            out = tf.warp_affine(
                out, translation_mat, dsize=(H, W), mode="bilinear", padding_mode="reflection", align_corners=False
            )

        # Scaling (after translation)
        if scale_factor is not None:
            # Make sure scale_factor is (B, 2)
            if scale_factor.dim() == 1:
                scale = scale_factor.unsqueeze(-1).repeat(1, 2)
            else:
                scale = scale_factor
            # Centered scaling: angle=0, center=same as before
            scale_mat = tf.get_rotation_matrix2d(center, torch.zeros(B, device=device), scale)
            out = tf.warp_affine(
                out, scale_mat, dsize=(H, W), mode="bilinear", padding_mode="reflection", align_corners=False
            )

        return out


class Transformer__affine(object):
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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(0.25)

        # match dimention
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


class ReverseResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super(ReverseResidualBlock, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size, 1, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(0.25)

        # Match dimension
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


class ResidualConvDecoder(nn.Module):
    def __init__(self, latent_dims, pixel):
        super(ResidualConvDecoder, self).__init__()
        self.res1 = ReverseResidualBlock(latent_dims, 512, 3, 2, 0, 1)
        self.res2 = ReverseResidualBlock(512, 128, 5, 2, 2, 0)
        self.res3 = ReverseResidualBlock(128, 64, 5, 2, 2, 0)
        self.res4 = ReverseResidualBlock(64, 64, 3, 2, 1, 0)
        self.res5 = ReverseResidualBlock(64, 32, 5, 2, 2, 1)
        self.res6 = ReverseResidualBlock(32, 32, 3, 2, 1, 1)
        self.rev_conv = nn.ConvTranspose2d(32, 1, 3, 2, 1, 1)
        self.upsample = nn.Upsample(size=(pixel, pixel), mode="bilinear", align_corners=False)
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
        x = self.upsample(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, latent_dims, pixel, decoder_type="fc"):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(latent_dims, pixel)
        if decoder_type == "fc":
            self.decoder = Decoder(latent_dims, pixel)
        elif decoder_type == "conv":
            self.decoder = ResidualConvDecoder(latent_dims, pixel)
        else:
            raise ValueError("Unsupported decoder type: {}".format(decoder_type))
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
    def __init__(self, latent_dims, pixel, decoder_type="fc"):
        super(Siamese, self).__init__()
        self.autoencoder = AutoEncoder(latent_dims, pixel, decoder_type=decoder_type)
        self.transform = Transformer()
        self.pixel = pixel
        self.latent_dims = latent_dims
        self.decoder_type = decoder_type

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


def get_instance_model_optimizer(device, learning_rate=0.0001, z_dims=2, pixel=64, *, decoder_type="fc"):
    print("Loading model")
    model = Siamese(latent_dims=z_dims, pixel=pixel, decoder_type=decoder_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer

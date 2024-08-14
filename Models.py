import torch
import torch.nn as nn


# nc = 1
# image_size = 64
#
# features_d = 64
# features_g = 64
# Z_dim = 200



class Critic(nn.Module):
    def __init__(self, nc, image_size, features_d):
        super(Critic, self).__init__()
        self.nc = nc
        self.image_size = image_size

        self.l1 = nn.Linear(150, image_size * image_size * nc)
        self.disc = nn.Sequential(
            nn.Conv2d(nc * 2, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=1, padding=0),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, img, points21):
        x1 = img
        x2 = self.l1(points21)
        x2 = x2.reshape(-1, self.nc, self.image_size, self.image_size)
        combine = torch.cat((x1, x2), dim=1)
        return self.disc(combine)

# class Critic(nn.Module):
#     def __init__(self, ngpu):
#         super(Critic, self).__init__()
#         self.ngpu = ngpu
#         self.image_size = image_size
#         self.l1 = nn.Linear(100, image_size * image_size * nc)
#         self.disc = nn.Sequential(
#             nn.Conv2d(nc * 2, features_d, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2),
#             self._block(features_d, features_d * 2, 4, 2, 1),
#             self._block(features_d * 2, features_d * 4, 4, 2, 1),
#             self._block(features_d * 4, features_d * 8, 4, 2, 1),
#             nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=1, padding=0),
#             # nn.Sigmoid(),
#         )
#
#     def _block(self, in_channels, out_channels, kernel_size, stride, padding):
#         return nn.Sequential(
#             nn.Conv2d(
#                 in_channels, out_channels, kernel_size, stride, padding, bias=False,
#             ),
#             nn.InstanceNorm2d(out_channels, affine=True),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#
#     def forward(self, img, points21):
#         x1 = img
#         x2 = self.l1(points21)
#         # x2 = x2.reshape(int(b_size / ngpu), nc, image_size, image_size)
#         x2 = x2.reshape(-1, nc, image_size, image_size)
#         combine = torch.cat((x1, x2), dim=1)
#         return self.disc(combine)


class Generator(nn.Module):
    def __init__(self, channels_noise, z_dim1, z_dim2, nc, features_g):
        super(Generator, self).__init__()

        total_input_dim = z_dim1 + z_dim2 + channels_noise

        self.net = nn.Sequential(
            # First block (latent vector input)
            self._block(total_input_dim, features_g * 16, 4, 1, 0),  # 4x4

            # Second block
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # 8x8

            # Third block
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # 16x16

            # Fourth block
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # 32x32

            # Dropout Layer (Added to be controlled by DQN)
            nn.Dropout(p=0.4),  # Initial dropout rate can be modified

            # Output block
            nn.ConvTranspose2d(features_g * 2, nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # Output image 64x64
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, points,z1,z2):
        x = torch.cat((points,z1,z2),dim=1)

        return self.net(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

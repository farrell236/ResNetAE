from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class ResNetAE(object):

    def __init__(self,
                 n_ResidualBlock=8,
                 n_levels=4,
                 z_dim=10,
                 output_channels=1,
                 bUseMultiResSkips=True):

        self.n_ResidualBlock = n_ResidualBlock
        self.n_levels = n_levels
        self.max_filters = 2 ** (n_levels+3)
        self.z_dim = z_dim
        self.bUseMultiResSkips = bUseMultiResSkips
        self.output_channels = output_channels

    def ResidualBlock(self, x, in_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1)):
        """
        Full pre-activation ResNet Residual block
        https://arxiv.org/pdf/1603.05027.pdf
        """
        skip = x
        x = nn.BatchNorm2d(in_channels)(x)
        x = nn.ReLU()(x)
        x = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=1)(x)
        x = nn.BatchNorm2d(out_channels)(x)
        x = nn.ReLU()(x)
        x = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=1)(x)
        x = x + skip
        return x

    def encoder(self, x):
        """
        'Striving for simplicity: The all convolutional net'
        arXiv: https://arxiv.org/pdf/1412.6806.pdf
        'We find that max-pooling can simply be replaced by a convolutional layer
        with increased stride without loss in accuracy on several image recognition benchmarks'
        """

        x = nn.Conv2d(in_channels=3, out_channels=8,
                      kernel_size=(3, 3), stride=(1, 1), padding=1)(x)
        x = nn.ReLU()(x)

        skips = []

        for i in range(self.n_levels):

            n_filters_1 = 2 ** (i + 3)
            n_filters_2 = 2 ** (i + 4)
            ks = 2 ** (self.n_levels - i)

            if self.bUseMultiResSkips:
                _x = nn.Conv2d(in_channels=n_filters_1, out_channels=self.max_filters,
                               kernel_size=(ks, ks), stride=(ks, ks))(x)
                _x = nn.ReLU()(_x)
                skips.append(_x)

            for _ in range(self.n_ResidualBlock):
                x = self.ResidualBlock(x, in_channels=n_filters_1, out_channels=n_filters_1)

            x = nn.Conv2d(in_channels=n_filters_1, out_channels=n_filters_2,
                          kernel_size=(2, 2), stride=(2, 2), padding=0)(x)
            x = nn.ReLU()(x)

        if self.bUseMultiResSkips:
            x = sum([x] + skips)

        x = nn.Conv2d(in_channels=n_filters_2, out_channels=self.z_dim,
                      kernel_size=(3, 3), stride=(1, 1), padding=1)(x)
        x = nn.ReLU()(x)

        return x

    def decoder(self, z):

        z = z_top = nn.ReLU()(
            nn.Conv2d(in_channels=self.z_dim, out_channels=self.max_filters,
                      kernel_size=(3, 3), stride=(1, 1), padding=1)(z))

        for i in range(self.n_levels):

            n_filters_0 = 2 ** (self.n_levels - i + 3)
            n_filters_1 = 2 ** (self.n_levels - i + 2)
            ks = 2 ** (i+1)

            z = nn.ConvTranspose2d(in_channels=n_filters_0, out_channels=n_filters_1,
                                   kernel_size=(2, 2), stride=(2, 2))(z)
            z = nn.ReLU()(z)

            for _ in range(self.n_ResidualBlock):
                z = self.ResidualBlock(z, in_channels=n_filters_1, out_channels=n_filters_1)

            if self.bUseMultiResSkips:
                _z = nn.ConvTranspose2d(in_channels=self.max_filters, out_channels=n_filters_1,
                                        kernel_size=(ks, ks), stride=(ks, ks))(z_top)
                _z = nn.ReLU()(_z)
                z = z + _z

        z = nn.Conv2d(in_channels=n_filters_1, out_channels=self.output_channels,
                      kernel_size=(3, 3), stride=(1, 1), padding=1)(z)
        z = nn.ReLU()(z)

        return z


if __name__ == '__main__':

    model = ResNetAE(n_ResidualBlock=8,
                     n_levels=4,
                     z_dim=10,
                     output_channels=3,
                     bUseMultiResSkips=True)

    test_input = torch.rand(10, 3, 256, 256)

    z = model.encoder(test_input)
    out = model.decoder(z)

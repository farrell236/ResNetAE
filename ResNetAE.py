from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


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

    def ResidualBlock(self, x, filters=64, kernel_size=(3, 3), strides=(1, 1)):
        """
        Full pre-activation ResNet Residual block
        https://arxiv.org/pdf/1603.05027.pdf
        """
        skip = x
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.backend.relu(x)
        x = tf.keras.layers.Conv2D(kernel_size=kernel_size, filters=filters,
                                   strides=strides, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.backend.relu(x)
        x = tf.keras.layers.Conv2D(kernel_size=kernel_size, filters=filters,
                                   strides=strides, padding='same')(x)
        x = x + skip
        return x

    def encoder(self, x):
        """
        'Striving for simplicity: The all convolutional net'
        arXiv: https://arxiv.org/pdf/1412.6806.pdf
        'We find that max-pooling can simply be replaced by a convolutional layer
        with increased stride without loss in accuracy on several image recognition benchmarks'
        """

        x = tf.keras.layers.Conv2D(filters=8,
                                   kernel_size=(3, 3), strides=(1, 1),
                                   padding='same', activation=tf.nn.relu)(x)

        skips = []

        for i in range(self.n_levels):

            n_filters_1 = 2 ** (i + 3)
            n_filters_2 = 2 ** (i + 4)
            ks = 2 ** (self.n_levels - i)

            if self.bUseMultiResSkips:
                skips.append(tf.keras.layers.Conv2D(filters=self.max_filters,
                                                    kernel_size=(ks, ks), strides=(ks, ks),
                                                    padding='same', activation=tf.nn.relu)(x))

            for _ in range(self.n_ResidualBlock):
                x = self.ResidualBlock(x, filters=n_filters_1)

            x = tf.keras.layers.Conv2D(filters=n_filters_2,
                                       kernel_size=(2, 2), strides=(2, 2),
                                       padding='same', activation=tf.nn.relu)(x)

        if self.bUseMultiResSkips:
            x = tf.add_n([x] + skips)

        x = tf.keras.layers.Conv2D(filters=self.z_dim,
                                   kernel_size=(3, 3), strides=(1, 1),
                                   padding='same', activation=tf.nn.relu)(x)

        return x

    def decoder(self, z):

        z = z_top = tf.keras.layers.Conv2D(filters=self.max_filters,
                                           kernel_size=(3, 3), strides=(1, 1),
                                           padding='same', activation=tf.nn.relu)(z)

        for i in range(self.n_levels):

            n_filters = 2 ** (self.n_levels - i + 2)
            ks = 2 ** (i+1)

            z = tf.keras.layers.Conv2DTranspose(filters=n_filters,
                                                kernel_size=(2, 2), strides=(2, 2),
                                                padding='same', activation=tf.nn.relu)(z)

            for _ in range(self.n_ResidualBlock):
                z = self.ResidualBlock(z, filters=n_filters)

            if self.bUseMultiResSkips:
                z += tf.keras.layers.Conv2DTranspose(filters=n_filters,
                                                     kernel_size=(ks, ks), strides=(ks, ks),
                                                     padding='same', activation=tf.nn.relu)(z_top)

        z = tf.keras.layers.Conv2D(filters=self.output_channels,
                                   kernel_size=(3, 3), strides=(1, 1),
                                   padding='same', activation=tf.nn.relu)(z)

        return z


if __name__ == '__main__':

    model = ResNetAE(n_ResidualBlock=8,
                     n_levels=4,
                     z_dim=10,
                     output_channels=3,
                     bUseMultiResSkips=True)

    x_in = tf.keras.layers.Input(shape=(None, None, 3))
    z = model.encoder(x_in)
    x_out = model.decoder(z)

    ResNetAE_model = tf.keras.models.Model(inputs=x_in, outputs=x_out)
    ResNetAE_model.summary()

    test_input = tf.random.normal(shape=(10, 32, 32, 3), dtype=tf.float32)

    out = ResNetAE_model(test_input)

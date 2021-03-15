from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), stride=(1, 1)):

        super(ResidualBlock, self).__init__()

        self.residual_block = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                   strides=stride, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                   strides=stride, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2),
        ])

    def call(self, x, **kwargs):
        return x + self.residual_block(x)


class ResNetEncoder(tf.keras.models.Model):
    def __init__(self,
                 n_ResidualBlock=8,
                 n_levels=4,
                 z_dim=10,
                 bUseMultiResSkips=True):

        super(ResNetEncoder, self).__init__()

        self.max_filters = 2 ** (n_levels+3)
        self.n_levels = n_levels
        self.bUseMultiResSkips = bUseMultiResSkips

        self.conv_list = []
        self.res_blk_list = []
        self.multi_res_skip_list = []

        self.input_conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3),
                                   strides=(1, 1), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2),
        ])

        for i in range(n_levels):
            n_filters_1 = 2 ** (i + 3)
            n_filters_2 = 2 ** (i + 4)
            ks = 2 ** (n_levels - i)

            self.res_blk_list.append(
                tf.keras.Sequential([ResidualBlock(n_filters_1)
                                     for _ in range(n_ResidualBlock)])
            )

            self.conv_list.append(
                tf.keras.Sequential([
                    tf.keras.layers.Conv2D(filters=n_filters_2, kernel_size=(2, 2),
                                           strides=(2, 2), padding='same'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(alpha=0.2),
                ])
            )

            if bUseMultiResSkips:
                self.multi_res_skip_list.append(
                    tf.keras.Sequential([
                        tf.keras.layers.Conv2D(filters=self.max_filters, kernel_size=(ks, ks),
                                               strides=(ks, ks), padding='same'),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU(alpha=0.2),
                    ])
                )

        self.output_conv = tf.keras.layers.Conv2D(filters=z_dim, kernel_size=(3, 3),
                                                  strides=(1, 1), padding='same')

    def call(self, x, **kwargs):

        x = self.input_conv(x)

        skips = []
        for i in range(self.n_levels):
            x = self.res_blk_list[i](x)
            if self.bUseMultiResSkips:
                skips.append(self.multi_res_skip_list[i](x))
            x = self.conv_list[i](x)

        if self.bUseMultiResSkips:
            x = sum([x] + skips)

        x = self.output_conv(x)

        return x


class ResNetDecoder(tf.keras.models.Model):
    def __init__(self,
                 n_ResidualBlock=8,
                 n_levels=4,
                 output_channels=3,
                 bUseMultiResSkips=True):

        super(ResNetDecoder, self).__init__()

        self.max_filters = 2 ** (n_levels+3)
        self.n_levels = n_levels
        self.bUseMultiResSkips = bUseMultiResSkips

        self.conv_list = []
        self.res_blk_list = []
        self.multi_res_skip_list = []

        self.input_conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=self.max_filters, kernel_size=(3, 3),
                                   strides=(1, 1), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2),
        ])

        for i in range(n_levels):
            n_filters = 2 ** (self.n_levels - i + 2)
            ks = 2 ** (i + 1)

            self.res_blk_list.append(
                tf.keras.Sequential([ResidualBlock(n_filters)
                                     for _ in range(n_ResidualBlock)])
            )

            self.conv_list.append(
                tf.keras.Sequential([
                    tf.keras.layers.Conv2DTranspose(filters=n_filters, kernel_size=(2, 2),
                                                    strides=(2, 2), padding='same'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(alpha=0.2),
                ])
            )

            if bUseMultiResSkips:
                self.multi_res_skip_list.append(
                    tf.keras.Sequential([
                        tf.keras.layers.Conv2DTranspose(filters=n_filters, kernel_size=(ks, ks),
                                                        strides=(ks, ks), padding='same'),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU(alpha=0.2),
                    ])
                )

        self.output_conv = tf.keras.layers.Conv2D(filters=output_channels, kernel_size=(3, 3),
                                                  strides=(1, 1), padding='same')

    def call(self, z, **kwargs):

        z = z_top = self.input_conv(z)

        for i in range(self.n_levels):
            z = self.conv_list[i](z)
            z = self.res_blk_list[i](z)
            if self.bUseMultiResSkips:
                z += self.multi_res_skip_list[i](z_top)

        z = self.output_conv(z)

        return z


class ResNetAE(tf.keras.models.Model):
    def __init__(self,
                 input_shape=(256, 256, 3),
                 n_ResidualBlock=8,
                 n_levels=4,
                 z_dim=128,
                 bottleneck_dim=128,
                 bUseMultiResSkips=True):
        super(ResNetAE, self).__init__()

        assert input_shape[0] == input_shape[1]
        output_channels = input_shape[2]
        self.z_dim = z_dim
        self.img_latent_dim = input_shape[0] // (2 ** n_levels)

        self.encoder = ResNetEncoder(n_ResidualBlock=n_ResidualBlock, n_levels=n_levels,
                                     z_dim=z_dim, bUseMultiResSkips=bUseMultiResSkips)
        self.decoder = ResNetDecoder(n_ResidualBlock=n_ResidualBlock, n_levels=n_levels,
                                     output_channels=output_channels, bUseMultiResSkips=bUseMultiResSkips)

        self.fc1 = tf.keras.layers.Dense(bottleneck_dim)
        self.fc2 = tf.keras.layers.Dense(self.img_latent_dim * self.img_latent_dim * self.z_dim)

    def encode(self, x):
        h = self.encoder(x)
        h = tf.keras.backend.reshape(h, shape=(-1, self.img_latent_dim * self.img_latent_dim * self.z_dim))
        return self.fc1(h)

    def decode(self, z):
        z = self.fc2(z)
        z = tf.keras.backend.reshape(z, shape=(-1, self.img_latent_dim, self.img_latent_dim, self.z_dim))
        h = self.decoder(z)
        return tf.keras.backend.sigmoid(h)

    def call(self, x, **kwargs):
        return self.decode(self.encode(x))


class ResNetVAE(tf.keras.models.Model):
    def __init__(self,
                 input_shape=(256, 256, 3),
                 n_ResidualBlock=8,
                 n_levels=4,
                 z_dim=128,
                 bottleneck_dim=128,
                 bUseMultiResSkips=True):
        super(ResNetVAE, self).__init__()

        assert input_shape[0] == input_shape[1]
        output_channels = input_shape[2]
        self.z_dim = z_dim
        self.img_latent_dim = input_shape[0] // (2 ** n_levels)

        self.encoder = ResNetEncoder(n_ResidualBlock=n_ResidualBlock, n_levels=n_levels,
                                     z_dim=z_dim, bUseMultiResSkips=bUseMultiResSkips)
        self.decoder = ResNetDecoder(n_ResidualBlock=n_ResidualBlock, n_levels=n_levels,
                                     output_channels=output_channels, bUseMultiResSkips=bUseMultiResSkips)


        # Assumes the input to be of shape 256x256
        self.fc21 = tf.keras.layers.Dense(bottleneck_dim)
        self.fc22 = tf.keras.layers.Dense(bottleneck_dim)
        self.fc3 = tf.keras.layers.Dense(self.img_latent_dim * self.img_latent_dim * self.z_dim)

    def encode(self, x):
        h1 = self.encoder(x)
        h1 = tf.keras.backend.reshape(h1, shape=(-1, self.img_latent_dim * self.img_latent_dim * self.z_dim))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = tf.keras.backend.exp(0.5*logvar)
        eps = tf.random.normal(std.shape)
        return mu + eps*std

    def decode(self, z):
        z = self.fc3(z)
        z = tf.keras.backend.reshape(z, shape=(-1, self.img_latent_dim, self.img_latent_dim, self.z_dim))
        h3 = self.decoder(z)
        return tf.keras.backend.sigmoid(h3)

    def call(self, x, **kwargs):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VectorQuantizer(tf.keras.layers.Layer):
    """
    Implementation of VectorQuantizer Layer from: simplegan.autoencoder.vq_vae
    url: https://simplegan.readthedocs.io/en/latest/_modules/simplegan/autoencoder/vq_vae.html
    """
    def __init__(self, num_embeddings, embedding_dim, commiment_cost):
        super(VectorQuantizer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commiment_cost = commiment_cost

        initializer = tf.keras.initializers.VarianceScaling(distribution='uniform')
        self.embedding = tf.Variable(
            initializer(shape=[self.embedding_dim, self.num_embeddings]), trainable=True
        )

    def call(self, x, **kwargs):

        flat_x = tf.reshape(x, [-1, self.embedding_dim])

        distances = (
            tf.math.reduce_sum(flat_x ** 2, axis=1, keepdims=True)
            - 2 * tf.linalg.matmul(flat_x, self.embedding)
            + tf.math.reduce_sum(self.embedding ** 2, axis=0, keepdims=True)
        )

        encoding_indices = tf.math.argmax(-distances, axis=1)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        encoding_indices = tf.reshape(encoding_indices, tf.shape(x)[:-1])
        quantized = tf.linalg.matmul(encodings, tf.transpose(self.embedding))
        quantized = tf.reshape(quantized, x.shape)

        e_latent_loss = tf.math.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        q_latent_loss = tf.math.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)

        loss = q_latent_loss + self.commiment_cost * e_latent_loss

        quantized = x + tf.stop_gradient(quantized - x)
        avg_probs = tf.math.reduce_mean(encodings, axis=0)
        perplexity = tf.math.exp(
            -tf.math.reduce_sum(avg_probs * tf.math.log(avg_probs + 1e-10))
        )

        return loss, quantized, perplexity, encoding_indices

    def quantize_encoding(self, x):
        encoding_indices = tf.keras.backend.flatten(x)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.linalg.matmul(encodings, tf.transpose(self.embedding))
        quantized = tf.reshape(quantized, [-1] + x.shape[1:].as_list() + [self.embedding_dim])
        return quantized


class ResNetVQVAE(tf.keras.models.Model):
    def __init__(self,
                 input_shape=(256, 256, 3),
                 n_ResidualBlock=8,
                 n_levels=4,
                 z_dim=128,
                 vq_num_embeddings=512,
                 vq_embedding_dim=64,
                 vq_commiment_cost=0.25,
                 bUseMultiResSkips=True):
        super(ResNetVQVAE, self).__init__()

        assert input_shape[0] == input_shape[1]
        output_channels = input_shape[2]
        self.z_dim = z_dim
        self.img_latent_dim = input_shape[0] // (2 ** n_levels)

        self.encoder = ResNetEncoder(n_ResidualBlock=n_ResidualBlock, n_levels=n_levels,
                                     z_dim=z_dim, bUseMultiResSkips=bUseMultiResSkips)
        self.decoder = ResNetDecoder(n_ResidualBlock=n_ResidualBlock, n_levels=n_levels,
                                     output_channels=output_channels, bUseMultiResSkips=bUseMultiResSkips)

        self.vq_vae = VectorQuantizer(num_embeddings=vq_num_embeddings,
                                      embedding_dim=vq_embedding_dim,
                                      commiment_cost=vq_commiment_cost)
        self.pre_vq_conv = tf.keras.layers.Conv2D(vq_embedding_dim, kernel_size=(1, 1), strides=(1, 1))

    def call(self, x, **kwargs):
        x = self.encoder(x)
        x = self.pre_vq_conv(x)
        loss, quantized, perplexity, encodings = self.vq_vae(x)
        x_recon = self.decoder(quantized)
        return loss, x_recon, perplexity, encodings


if __name__ == '__main__':

    import numpy as np

    # encoder = ResNetEncoder()
    # decoder = ResNetDecoder()
    #
    # out_encoder = encoder(np.random.rand(10, 256, 256, 3).astype('float32'))
    # out_decoder = decoder(np.random.rand(10, 16, 16, 10).astype('float32'))

    a=1

    ae = ResNetAE()
    out = ae(np.random.rand(10, 256, 256, 3).astype('float32'))

    a=1

    vae = ResNetVAE()
    out = vae(np.random.rand(10, 256, 256, 3).astype('float32'))

    a=1

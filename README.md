# ResNet Auto-Encoder

This repository contains Tensorflow code for an Auto-Encoder architecture built with Residual Blocks. 

## Default Architecture Parameters:

```
model = ResNetAE(n_ResidualBlock=8,       
                 n_levels=4,             
                 z_dim=10,
                 output_channels=1,
                 bUseMultiResSkips=True,
                 is_training=True)
```

- ```n_ResidualBlock```: Number of Convolutional residual blocks at each resolution
- ```n_levels```: Number of scaling resolutions, at each increased resolution, the image dimension halves and the number of filters channel doubles
- ```z_dim```: Number of latent dim filters
- ```output_channels```: Number of output channels (1: greyscale, 3: RGB, n: user defined)
- ```bUseMultiResSkips```: At each resolution, the feature maps are added to the latent/image output (green path in diagram)
- ```is_training```: Default is true, this parameter is passed into ```tf.layers.batch_normalization``` only.



## Encoder

<p align="center">
  <img src="https://github.com/farrell236/ResNetAE/blob/master/architecture/encoder.png" alt="ResNetAE Encoder">
</p>

The encoder expects a 4-D Image Tensor in the form of ```[Batch x Height x Width x Channels]```. The output ```z``` would be of shape ```[Batch x Height/n_levels x Width/n_levels x z_dim]```.

```
with tf.variable_scope('encoder'):
    z = model.encoder(tf.cast(x_in,tf.float32))
```

N.B. It is possible to flatten ```z``` by ```tf.layers.dense``` for a vectorised latent space, as long as the shape is preserved for the decoder during the unflatten process.


## Decoder

<p align="center">
  <img src="https://github.com/farrell236/ResNetAE/blob/master/architecture/decoder.png" alt="ResNetAE Decoder">
</p>

The decoder expects a 4-D Feature Tensor in the form of ```[Batch x Height x Width x Channels]```. The output ```x_out``` would be of the shape ```[Batch x Height*n_levels x Width*n_levels x output_channels]```

```
with tf.variable_scope('decoder'):
    x_out = model.decoder(z)
```


## Residual Block

The Residual Block uses the [Full pre-activation ResNet Residual block](https://arxiv.org/pdf/1603.05027.pdf) by He et al.

<p align="center">
  <img src="https://github.com/farrell236/ResNetAE/blob/master/architecture/residual_block.png" alt="ResNetAE Residual Block">
</p>





# ResNet Auto-Encoder

This repository contains Tensorflow code for an Auto-Encoder architecture built with Residual Blocks. 

## Default Architecture Parameters:

```
model = ResNetAE(input_shape=(256, 256, 3),
                 n_ResidualBlock=8,
                 n_levels=4,
                 z_dim=128,
                 bottleneck_dim=128,
                 bUseMultiResSkips=True)
```

- ```input_shape```: A tuple defining the input image shape for the model
- ```n_ResidualBlock```: Number of Convolutional residual blocks at each resolution
- ```n_levels```: Number of scaling resolutions, at each increased resolution, the image dimension halves and the number of filters channel doubles
- ```z_dim```: Number of latent dim filters
- ```bottleneck_dim```: AE/VAE vectorized latent space dimension 
- ```bUseMultiResSkips```: At each resolution, the feature maps are added to the latent/image output (green path in diagram)


## Encoder

<p align="center">
  <img src="https://github.com/farrell236/ResNetAE/blob/master/architecture/encoder.png" alt="ResNetAE Encoder">
</p>

The encoder expects a 4-D Image Tensor in the form of ```[Batch x Height x Width x Channels]```. The output ```z``` would be of shape ```[Batch x Height/(2**n_levels) x Width/(2**n_levels) x z_dim]```.

```
encoder = ResNetEncoder(n_ResidualBlock=8, 
                        n_levels=4,
                        z_dim=10, 
                        bUseMultiResSkips=True)
```

N.B. It is possible to flatten ```z``` by ```tf.layers.dense``` for a vectorised latent space, as long as the shape is preserved for the decoder during the unflatten process.


## Decoder

<p align="center">
  <img src="https://github.com/farrell236/ResNetAE/blob/master/architecture/decoder.png" alt="ResNetAE Decoder">
</p>

The decoder expects a 4-D Feature Tensor in the form of ```[Batch x Height x Width x Channels]```. The output ```x_out``` would be of the shape ```[Batch x Height*(2**n_levels) x Width*(2**n_levels) x output_channels]```

```
decoder = ResNetDecoder(n_ResidualBlock=8, 
                        n_levels=4,
                        output_channels=3, 
                        bUseMultiResSkips=True)
```


## Residual Block

The Residual Block uses the [Full pre-activation ResNet Residual block](https://arxiv.org/pdf/1603.05027.pdf) by He et al.

TODO: implementation changed to Conv-Batch-Relu, update figure

<p align="center">
  <img src="https://github.com/farrell236/ResNetAE/blob/master/architecture/residual_block.png" alt="ResNetAE Residual Block">
</p>


## 

If you find this work useful for your research, please cite:

```
@article{ResNetAE, 
  Title={{R}es{N}et{AE}-https://github.com/farrell236/ResNetAE}, 
  url={https://github.com/farrell236/ResNetAE},  
  Author={Hou, Benjamin}, 
  Year={2019}
}
```


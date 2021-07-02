from typing import Any

import flax.linen as nn
import jax.numpy as jnp
from chex import Array
from einops import rearrange

Dtype = Any

__all__ = ["MlpBlock", "MixerBlock", "MlpMixer"]


class MlpBlock(nn.Module):
    """
    A MLPBlock Wrapper Flax Module consisting of two
    Fully connected layers with a GELU layer in between.

    Attributes:
        mlp_dim: No of output dimensions for the first FC
        approximate: If True, uses the approximate formulation of GELU
        dtype: the dtype of the computation (default: float32)
    """

    mlp_dim: int
    approximate: bool = True
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x) -> Array:
        y = nn.Dense(features=self.mlp_dim, dtype=self.dtype)(x)
        y = nn.gelu(y, approximate=self.approximate)
        out = nn.Dense(features=x.shape[-1], dtype=self.dtype)(y)
        return out


class MixerBlock(nn.Module):
    """
    A Flax Module to act as the mixer block layer for the MLP-Mixer Architecture.

    Attributes:
        tokens_mlp_dim: No of dimensions for the MLP Block 1
        channels_mlp_dim: No of dimensions for the MLP Block 2
        approximate: If True, uses the approximate formulation of GELU in each MLP Block
        dtype: the dtype of the computation (default: float32)
    """

    tokens_mlp_dim: int
    channels_mlp_dim: int
    approximate: bool = True
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x) -> Array:
        # Layer Normalization
        y = nn.LayerNorm(dtype=self.dtype)(x)
        # Transpose
        y = jnp.swapaxes(y, 1, 2)
        # MLP 1
        y = MlpBlock(
            mlp_dim=self.tokens_mlp_dim,
            approximate=self.approximate,
            dtype=self.dtype,
            name="token_mixing",
        )(y)
        # Transpose
        y = jnp.swapaxes(y, 1, 2)
        # Skip Connection
        x = x + y
        # Layer Normalization
        y = nn.LayerNorm(dtype=self.dtype)(x)
        # MLP 2 with Skip Connection
        out = x + MlpBlock(
            mlp_dim=self.channels_mlp_dim,
            approximate=self.approximate,
            dtype=self.dtype,
            name="channel_mixing",
        )(y)
        return out


class MlpMixer(nn.Module):
    """
    Flax Module for the MLP-Mixer Architecture.

    Attributes:
        patches: Patch configuration
        num_classes: No of classes for the output head
        num_blocks: No of Blocks of Mixers to use
        hidden_dim: No of Hidden Dimension for the Patch-Wise Convolution Layer
        tokens_mlp_dim: No of dimensions for the MLP Block 1
        channels_mlp_dim: No of dimensions for the MLP Block 2
        approximate: If True, uses the approximate formulation of GELU in each MLP Block
        dtype: the dtype of the computation (default: float32)
    """

    patches: Any
    num_classes: int
    num_blocks: int
    hidden_dim: int
    tokens_mlp_dim: int
    channels_mlp_dim: int
    approximate: bool = True
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, *, train) -> Array:
        del train
        # Per-Patch Fully Connected Layer
        x = nn.Conv(
            features=self.hidden_dim,
            kernel_size=self.patches["size"],
            strides=self.patches["size"],
            dtype=self.dtype,
            name="stem",
        )(inputs)
        x = rearrange(x, "n h w c -> n (h w) c")
        # Num Blocks x Mixer Blocks
        for _ in range(self.num_blocks):
            x = MixerBlock(
                tokens_mlp_dim=self.tokens_mlp_dim,
                channels_mlp_dim=self.channels_mlp_dim,
                approximate=self.approximate,
                dtype=self.dtype,
            )(x)
        # Output Head
        x = nn.LayerNorm(dtype=self.dtype, name="pre_head_layer_norm")(x)
        x = jnp.mean(x, axis=1, dtype=self.dtype)
        return nn.Dense(
            self.num_classes,
            kernel_init=nn.initializers.zeros,
            dtype=self.dtype,
            name="head",
        )(x)

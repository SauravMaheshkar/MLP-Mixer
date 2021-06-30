"""
Instance Checks for Various Modules
"""
import flax.linen as nn

from mlpmixer_flax.models import MixerBlock, MlpBlock, MlpMixer


def test_layers():

    block = MlpBlock(mlp_dim=128)
    mixer_block = MixerBlock(tokens_mlp_dim=128, channels_mlp_dim=128)
    model = MlpMixer(
        patches=16,
        num_classes=10,
        num_blocks=4,
        hidden_dim=512,
        tokens_mlp_dim=128,
        channels_mlp_dim=128,
    )

    assert isinstance(block, nn.Module)
    assert isinstance(mixer_block, nn.Module)
    assert isinstance(model, nn.Module)

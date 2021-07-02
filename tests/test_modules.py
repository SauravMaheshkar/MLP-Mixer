"""
Instance Checks for Various Modules
"""
import flax.linen as nn

from mlpmixer_flax.config import mixer_b16_config
from mlpmixer_flax.dataloader import get_dataset_info
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


def test_modelwithconfig():

    dataset = "cifar10"
    num_classes = get_dataset_info(dataset, "train")["num_classes"]
    model = MlpMixer(num_classes=num_classes, **mixer_b16_config)

    assert isinstance(model, nn.Module)

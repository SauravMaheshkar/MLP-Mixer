from typing import Dict

from mlpmixer_flax.config import configuration, mixer_b16_config


def test_config():

    assert isinstance(configuration, Dict)
    assert isinstance(mixer_b16_config, Dict)

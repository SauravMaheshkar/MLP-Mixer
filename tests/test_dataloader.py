from typing import Dict

from mlpmixer_flax.config import configuration
from mlpmixer_flax.dataloader import get_dataset_info


def test_dataloader():

    dataset = "cifar10"
    batch_size = 512
    config = configuration
    num_classes = get_dataset_info(dataset, "train")["num_classes"]
    config.update(batch=batch_size)
    config.update({"pp": {"crop": 224}})

    assert isinstance(num_classes, int)
    assert isinstance(config, Dict)

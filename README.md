# Flax implementation of MLP-Mixer from "MLP-Mixer: An all-MLP Architecture for Vision"

![](https://github.com/SauravMaheshkar/MLP-Mixer/blob/main/assets/MLPMixer%20Banner.png?raw=true)

The strong performance of recent vision architectures is often attributed to Attention or Convolutions. But Multi Layer Perceptrons have always been better at capturing long-range dependencies and positional patterns, but admittedly fall behind when it comes to learning local features, which is where CNNs shine. An interesting new perspective of viewing convolutions as a "**sparse FC with shared parameters**" was proposed in [**Ding, et al**](https://arxiv.org/pdf/2105.01883.pdf). This perspective opens up a new way of looking at architectures. In this report we'll look at one such [**paper**](https://arxiv.org/pdf/2105.01601v1.pdf) which explores the idea of using convolutions with an extremely small kernel size of (1,1) essentially turning convolutions into standard matrix multiplications applied independently to each spatial location.  This modification alone doesn't allow for the aggregation of spatial information. To compensate for this the authors proposed dense matrix multiplications that are applied to every feature across all spatial locations.

This repository includes a minimal implementation of MLP-Mixer written in [Flax](https://github.com/google/flax). Most of the codebase is ported from the [original implementation](https://github.com/google-research/vision_transformer).

## Installation

You can install this package from PyPI:

```sh
pip install mlpmixer-flax
```

Or directly from GitHub:

```sh
pip install --upgrade git+https://github.com/SauravMaheshkar/MLP-Mixer.git
```

## Usage

```python
from mlpmixer_flax.config import mixer_b16_config
from mlpmixer_flax.dataloader import get_dataset_info
from mlpmixer_flax.models import MlpMixer

dataset = "cifar10"
num_classes = get_dataset_info(dataset, "train")["num_classes"]
model = MlpMixer(num_classes=num_classes, **mixer_b16_config)
```

The easiest way to get started would be to try out the [FineTuning Example Notebook](https://github.com/SauravMaheshkar/MLP-Mixer/blob/main/examples/FineTuning_Example.ipynb).

## Development

### 1. Conda Approach

```sh
conda env create --name <env-name> sauravmaheshkar/mlpmixer
conda activate <env-name>
```

### 2. Docker Approach

```sh
docker pull ghcr.io/sauravmaheshkar/mlpmixer-dev:latest
docker run -it -d --name <container_name> ghcr.io/sauravmaheshkar/mlpmixer-dev
```

Use the [Remote Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) Extension in VSCode and [attach to the running container](https://code.visualstudio.com/docs/remote/attach-container). The code resides in the `code/` dir.

Alternatively you can also download the image from [Docker Hub](https://hub.docker.com/r/sauravmaheshkar/mlpmixer-dev).

```sh
docker pull sauravmaheshkar/mlpmixer-dev
```

## Citations

```bibtex
@misc{tolstikhin2021mlpmixer,
    title   = {MLP-Mixer: An all-MLP Architecture for Vision},
    author  = {Ilya Tolstikhin and Neil Houlsby and Alexander Kolesnikov and Lucas Beyer and Xiaohua Zhai and Thomas Unterthiner and Jessica Yung and Daniel Keysers and Jakob Uszkoreit and Mario Lucic and Alexey Dosovitskiy},
    year    = {2021},
    eprint  = {2105.01601},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

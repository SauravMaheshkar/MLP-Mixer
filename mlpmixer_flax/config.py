__all__ = ["configuration", "with_dataset", "mixer_b16_config"]

from typing import Dict

configuration = {
    "pretrained_dir": ".",
    "tfds_data_dir": None,
    "total_steps": None,
    "tfds_manual_dir": None,
    "grad_norm_clip": 1.0,
    "optim_dtype": "float32",  # "bfloat16" or "float32"
    "accum_steps": 8,
    "batch": 512,
    "batch_eval": 512,
    "shuffle_buffer": 50000,
    "eval_every": 100,
    "progress_every": 10,
    "checkpoint_every": 1000,
    "prefetch": 2,
    "base_lr": 0.03,
    "decay_type": "cosine",  # "cosine" or "linear"
    "warmup_steps": 500,
    "trainer": "train",  # "train" or "inference_time"
    "model": None,
    "dataset": None,
    "pp": None,
}

DATASET_PRESETS = {
    "cifar10": dict(
        {
            "total_steps": 10000,
            "pp": dict({"train": "train[:98%]", "test": "test", "crop": 384}),
        }
    ),
    "cifar100": dict(
        {
            "total_steps": 10000,
            "pp": dict({"train": "train[:98%]", "test": "test", "crop": 384}),
        }
    ),
}

mixer_b16_config = {
    "name": "Mixer-B_16",
    "patches": {"size": (16, 16)},
    "hidden_dim": 768,
    "num_blocks": 12,
    "tokens_mlp_dim": 384,
    "channels_mlp_dim": 3072,
}


def with_dataset(config: Dict, dataset: str) -> Dict:
    config.update(dataset=dataset)
    config.update(DATASET_PRESETS[dataset])
    return config

# Ported from: https://github.com/google-research/vision_transformer/blob/master/vit_jax/input_pipeline.py
import sys

import flax
import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

if sys.platform != "darwin":
    # A workaround to avoid crash because tfds may open to many files.
    import resource

    low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

# Adjust depending on the available RAM.
MAX_IN_MEMORY = 200_000


def get_dataset_info(dataset, split):
    """Returns information about tfds dataset -- see `get_dataset_info()`."""
    data_builder = tfds.builder(dataset)
    return dict(
        num_examples=data_builder.info.splits[split].num_examples,
        num_classes=data_builder.info.features["label"].num_classes,
        int2str=data_builder.info.features["label"].int2str,
        examples_glob=None,
    )


def get_data_from_tfds(*, config, mode):
    """Returns dataset as read from tfds dataset `config.dataset`."""

    data_builder = tfds.builder(config["dataset"], data_dir=config["tfds_data_dir"])

    data_builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(
            manual_dir=config["tfds_manual_dir"]
        )
    )

    data = data_builder.as_dataset(
        split=config["cifar10"]["pp"][mode],
        # Reduces memory footprint in shuffle buffer.
        decoders={"image": tfds.decode.SkipDecoding()},
        shuffle_files=mode,
    )
    image_decoder = data_builder.info.features["image"].decode_example

    dataset_info = get_dataset_info(config["dataset"], config["cifar10"]["pp"][mode])
    return get_data(
        data=data,
        mode=mode,
        num_classes=dataset_info["num_classes"],
        image_decoder=image_decoder,
        repeats=None if mode == "train" else 1,
        batch_size=config["batch_eval"] if mode == "test" else config["batch"],
        image_size=config["pp"]["crop"],
        shuffle_buffer=min(dataset_info["num_examples"], config["shuffle_buffer"]),
    )


def get_data(
    *,
    data,
    mode,
    num_classes,
    image_decoder,
    repeats,
    batch_size,
    image_size,
    shuffle_buffer,
    preprocess=None,
):
    """Returns dataset for training/eval.
    Args:
        data: tf.data.Dataset to read data from.
        mode: Must be "train" or "test".
        num_classes: Number of classes (used for one-hot encoding).
        image_decoder: Applied to `features['image']` after shuffling. Decoding the
            image after shuffling allows for a larger shuffle buffer.
        repeats: How many times the dataset should be repeated. For indefinite
            repeats specify None.
        batch_size: Global batch size. Note that the returned dataset will have
            dimensions [local_devices, batch_size / local_devices, ...].
        image_size: Image size after cropping (for training) / resizing (for
            evaluation).
        shuffle_buffer: Number of elements to preload the shuffle buffer with.
        preprocess: Optional preprocess function. This function will be applied to
            the dataset just after repeat/shuffling, and before the data augmentation
            preprocess step is applied.
    """

    def _pp(data):
        im = image_decoder(data["image"])
        if mode == "train":
            channels = im.shape[-1]
            begin, size, _ = tf.image.sample_distorted_bounding_box(
                tf.shape(im),
                tf.zeros([0, 0, 4], tf.float32),
                area_range=(0.05, 1.0),
                min_object_covered=0,  # Don't enforce a minimum area.
                use_image_if_no_bounding_boxes=True,
            )
            im = tf.slice(im, begin, size)
            # Unfortunately, the above operation loses the depth-dimension. So we
            # need to restore it the manual way.
            im.set_shape([None, None, channels])
            im = tf.image.resize(im, [image_size, image_size])
            if tf.random.uniform(shape=[]) > 0.5:
                im = tf.image.flip_left_right(im)
        else:
            im = tf.image.resize(im, [image_size, image_size])
        im = (im - 127.5) / 127.5
        label = tf.one_hot(
            data["label"], num_classes
        )  # pylint: disable=no-value-for-parameter
        return {"image": im, "label": label}

    data = data.repeat(repeats)
    if mode == "train":
        data = data.shuffle(shuffle_buffer)
    if preprocess is not None:
        data = data.map(preprocess, tf.data.experimental.AUTOTUNE)
    data = data.map(_pp, tf.data.experimental.AUTOTUNE)
    data = data.batch(batch_size, drop_remainder=True)

    # Shard data such that it can be distributed accross devices
    num_devices = jax.local_device_count()

    def _shard(data):
        data["image"] = tf.reshape(
            data["image"], [num_devices, -1, image_size, image_size, 3]
        )
        data["label"] = tf.reshape(data["label"], [num_devices, -1, num_classes])
        return data

    if num_devices is not None:
        data = data.map(_shard, tf.data.experimental.AUTOTUNE)

    return data.prefetch(1)


def prefetch(dataset, n_prefetch):
    """Prefetches data to device and converts to numpy array."""
    ds_iter = iter(dataset)
    ds_iter = map(
        lambda x: jax.tree_map(lambda t: np.asarray(memoryview(t)), x), ds_iter
    )
    if n_prefetch:
        ds_iter = flax.jax_utils.prefetch_to_device(ds_iter, n_prefetch)
    return ds_iter

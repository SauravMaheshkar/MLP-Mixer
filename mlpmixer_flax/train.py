# From https://github.com/google-research/vision_transformer/blob/master/vit_jax/train.py
import jax
import jax.numpy as jnp

from .train_utils import accumulate_gradient


def make_update_fn(*, apply_fn, accum_steps, lr_fn):
    """Returns update step for data parallel training."""

    def update_fn(opt, step, batch, rng):

        _, new_rng = jax.random.split(rng)
        dropout_rng = jax.random.fold_in(rng, jax.lax.axis_index("batch"))

        def cross_entropy_loss(*, logits, labels):
            logp = jax.nn.log_softmax(logits)
            return -jnp.mean(jnp.sum(logp * labels, axis=1))

        def loss_fn(params, images, labels):
            logits = apply_fn(
                dict(params=params),
                rngs=dict(dropout=dropout_rng),
                inputs=images,
                train=True,
            )
            return cross_entropy_loss(logits=logits, labels=labels)

        l, g = accumulate_gradient(
            jax.value_and_grad(loss_fn),
            opt.target,
            batch["image"],
            batch["label"],
            accum_steps,
        )
        g = jax.tree_map(lambda x: jax.lax.pmean(x, axis_name="batch"), g)
        l = jax.lax.pmean(l, axis_name="batch")

        opt = opt.apply_gradient(g, learning_rate=lr_fn(step))
        return opt, l, new_rng

    return jax.pmap(update_fn, axis_name="batch", donate_argnums=(0,))

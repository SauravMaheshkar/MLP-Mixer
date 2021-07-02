# From https://github.com/google-research/vision_transformer/blob/master/vit_jax/utils.py
import flax
import jax
import jax.numpy as jnp
import numpy as np


class Optimizer(flax.optim.OptimizerDef):
    """Momentum optimizer that stores state using half-precision."""

    @flax.struct.dataclass
    class HyperParams:
        learning_rate: np.ndarray
        beta: np.ndarray
        grad_norm_clip: np.ndarray

    @flax.struct.dataclass
    class State:
        momentum: np.ndarray

    def __init__(
        self, learning_rate=None, beta=0.9, dtype="bfloat16", grad_norm_clip=None
    ):
        hyper_params = Optimizer.HyperParams(learning_rate, beta, grad_norm_clip)
        super().__init__(hyper_params)
        self.dtype = dict(bfloat16=jnp.bfloat16, float32=jnp.float32)[dtype]

    def init_param_state(self, param):
        return Optimizer.State(jnp.zeros_like(param, dtype=self.dtype))

    def apply_gradient(self, hyper_params, params, state, grads):
        step = state.step
        params_flat, treedef = jax.tree_flatten(params)
        states_flat = treedef.flatten_up_to(state.param_states)
        grads_flat = treedef.flatten_up_to(grads)

        # Optionally resize the global gradient to a maximum norm.
        if hyper_params.grad_norm_clip:
            grads_l2 = jnp.sqrt(sum([jnp.vdot(p, p) for p in grads_flat]))
            grads_factor = jnp.minimum(1.0, hyper_params.grad_norm_clip / grads_l2)
            grads_flat = jax.tree_map(lambda param: grads_factor * param, grads_flat)

        out = [
            self.apply_param_gradient(step, hyper_params, param, state, grad)
            for param, state, grad in zip(params_flat, states_flat, grads_flat)
        ]

        new_params_flat, new_states_flat = list(zip(*out)) if out else ((), ())
        new_params = jax.tree_unflatten(treedef, new_params_flat)
        new_param_states = jax.tree_unflatten(treedef, new_states_flat)
        new_state = flax.optim.OptimizerState(step + 1, new_param_states)
        return new_params, new_state

    def apply_param_gradient(self, step, hyper_params, param, state, grad):
        del step
        assert hyper_params.learning_rate is not None, "no learning rate provided."
        momentum = state.momentum
        new_momentum = hyper_params.beta * momentum + grad
        new_param = param - hyper_params.learning_rate * new_momentum
        new_state = Optimizer.State(new_momentum.astype(self.dtype))
        return new_param, new_state


def create_learning_rate_schedule(
    total_steps, base, decay_type, warmup_steps, linear_end=1e-5
):
    """Creates learning rate schedule.
    Currently only warmup + {linear,cosine} but will be a proper mini-language
    like preprocessing one in the future.
    Args:
        total_steps: The total number of steps to run.
        base: The starting learning-rate (without warmup).
        decay_type: 'linear' or 'cosine'.
        warmup_steps: how many steps to warm up for.
        linear_end: Minimum learning rate.
    Returns:
        A function learning_rate(step): float -> {"learning_rate": float}.
    """

    def step_fn(step):
        """Step to learning rate function."""
        lr = base

        progress = (step - warmup_steps) / float(total_steps - warmup_steps)
        progress = jnp.clip(progress, 0.0, 1.0)
        if decay_type == "linear":
            lr = linear_end + (lr - linear_end) * (1.0 - progress)
        elif decay_type == "cosine":
            lr = lr * 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
        else:
            raise ValueError(f"Unknown lr type {decay_type}")

        if warmup_steps:
            lr = lr * jnp.minimum(1.0, step / warmup_steps)

        return jnp.asarray(lr, dtype=jnp.float32)

    return step_fn


def accumulate_gradient(loss_and_grad_fn, params, images, labels, accum_steps):
    """Accumulate gradient over multiple steps to save on memory."""
    if accum_steps and accum_steps > 1:
        assert (
            images.shape[0] % accum_steps == 0
        ), f"Bad accum_steps {accum_steps} for batch size {images.shape[0]}"
        step_size = images.shape[0] // accum_steps
        l, g = loss_and_grad_fn(params, images[:step_size], labels[:step_size])

        def acc_grad_and_loss(i, l_and_g):
            imgs = jax.lax.dynamic_slice(
                images, (i * step_size, 0, 0, 0), (step_size,) + images.shape[1:]
            )
            lbls = jax.lax.dynamic_slice(
                labels, (i * step_size, 0), (step_size, labels.shape[1])
            )
            li, gi = loss_and_grad_fn(params, imgs, lbls)
            l, g = l_and_g
            return (l + li, jax.tree_multimap(lambda x, y: x + y, g, gi))

        l, g = jax.lax.fori_loop(1, accum_steps, acc_grad_and_loss, (l, g))
        return jax.tree_map(lambda x: x / accum_steps, (l, g))
    else:
        return loss_and_grad_fn(params, images, labels)

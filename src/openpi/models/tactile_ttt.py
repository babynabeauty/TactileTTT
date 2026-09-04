"""RoboTTT-style layer-wise fast-weight memory for tactile robot policies."""

from typing import TypeAlias

import flax.linen as nn
import jax
import jax.numpy as jnp

TactileTTTState: TypeAlias = tuple[jax.Array, jax.Array, jax.Array, jax.Array]


def tactile_contact_gate(
    contact_force: jax.Array,
    *,
    threshold: float,
    temperature: float,
) -> jax.Array:
    """Compute one soft write gate from unnormalized ``calc_force`` per item."""
    if contact_force.ndim != 4 or contact_force.shape[-1] != 3:
        raise ValueError(f"Expected contact_force [B,T,F,3], got {contact_force.shape}.")
    magnitude = jnp.linalg.norm(contact_force.astype(jnp.float32), axis=-1)
    score = jnp.max(magnitude, axis=(1, 2))
    temperature = jnp.asarray(max(float(temperature), 1e-6), dtype=jnp.float32)
    return jax.nn.sigmoid((score - float(threshold)) / temperature)


class LayerwiseTactileTTT(nn.Module):
    """Post-attention TTT-MLP for one action-expert Transformer block."""

    width: int
    memory_dim: int
    mlp_dim: int
    base_inner_lr: float
    residual_gate_init: float = 0.001

    @nn.compact
    def __call__(
        self,
        tokens: jax.Array,
        fast_state: TactileTTTState,
        write_gate: jax.Array,
        update_mask: jax.Array,
        write_tokens: jax.Array | None = None,
        target_tokens: jax.Array | None = None,
        layer_active: jax.Array | float = 1.0,
    ) -> tuple[jax.Array, TactileTTTState, dict[str, jax.Array]]:
        """Read memory for ``tokens`` and optionally learn temporal bindings.

        V1 leaves ``write_tokens``/``target_tokens`` unset and binds three
        projections of the same action-expert token sequence. V2 supplies
        adjacent tactile sequences: K comes from frames [0..T-2], V from
        frames [1..T-1], while Q still comes from the current expert tokens.
        """
        normalized = nn.LayerNorm(name="token_norm")(tokens)
        if write_tokens is None:
            if target_tokens is not None:
                raise ValueError("target_tokens requires write_tokens.")
            normalized_write = normalized
            normalized_target = normalized
        else:
            if target_tokens is None:
                raise ValueError("write_tokens requires target_tokens.")
            if write_tokens.shape[:2] != target_tokens.shape[:2]:
                raise ValueError(
                    "Temporal TTT write and target sequences must have equal batch/token dimensions; "
                    f"got {write_tokens.shape} and {target_tokens.shape}."
                )
            normalized_write = nn.LayerNorm(name="write_norm")(write_tokens)
            normalized_target = nn.LayerNorm(name="target_norm")(target_tokens)

        keys = nn.Dense(self.memory_dim, use_bias=False, name="key_proj")(normalized_write).astype(jnp.float32)
        values = nn.Dense(self.memory_dim, use_bias=False, name="value_proj")(normalized_target).astype(jnp.float32)
        queries = nn.Dense(self.memory_dim, use_bias=False, name="query_proj")(normalized).astype(jnp.float32)

        w1, b1, w2, b2 = (jnp.asarray(value, dtype=jnp.float32) for value in fast_state)

        def binding_loss(
            fw1: jax.Array,
            fb1: jax.Array,
            fw2: jax.Array,
            fb2: jax.Array,
        ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
            hidden = jax.nn.gelu(jnp.einsum("bnd,bdh->bnh", keys, fw1) + fb1[:, None, :])
            prediction = jnp.einsum("bnh,bhd->bnd", hidden, fw2) + fb2[:, None, :]
            error = prediction - values
            per_example = jnp.mean(jnp.square(error), axis=(1, 2))
            return jnp.sum(per_example), (per_example, prediction)

        (_, (reconstruction, _)), gradients = jax.value_and_grad(
            binding_loss,
            argnums=(0, 1, 2, 3),
            has_aux=True,
        )(w1, b1, w2, b2)

        inner_lr_raw = self.param("inner_lr_raw", nn.initializers.zeros_init(), ())
        inner_lr = self.base_inner_lr * jax.nn.softplus(inner_lr_raw) / jnp.log(2.0)
        update_mask = jnp.asarray(update_mask, dtype=jnp.float32)
        if update_mask.ndim == 0:
            update_mask = jnp.broadcast_to(update_mask, write_gate.shape)
        layer_active = jnp.asarray(layer_active, dtype=jnp.float32)
        if layer_active.ndim == 0:
            layer_active = jnp.broadcast_to(layer_active, write_gate.shape)
        effective_lr = inner_lr * write_gate.astype(jnp.float32) * update_mask * layer_active

        updated_state = (
            w1 - effective_lr[:, None, None] * gradients[0],
            b1 - effective_lr[:, None] * gradients[1],
            w2 - effective_lr[:, None, None] * gradients[2],
            b2 - effective_lr[:, None] * gradients[3],
        )

        updated_w1, updated_b1, updated_w2, updated_b2 = updated_state
        memory_hidden = jax.nn.gelu(jnp.einsum("bnd,bdh->bnh", queries, updated_w1) + updated_b1[:, None, :])
        memory = jnp.einsum("bnh,bhd->bnd", memory_hidden, updated_w2) + updated_b2[:, None, :]
        memory = nn.Dense(self.width, use_bias=False, name="output_proj")(memory.astype(tokens.dtype))
        # Linen Dense keeps float32 parameters by default, so its result may be
        # promoted back to float32 even when the activation is bfloat16.  The
        # Transformer blocks are scanned and require the hidden-state carry to
        # preserve its dtype exactly across every layer.
        memory = memory.astype(tokens.dtype)

        residual_gate_raw = self.param(
            "residual_gate",
            nn.initializers.constant(self.residual_gate_init),
            (self.width,),
        )
        residual_gate = jnp.tanh(residual_gate_raw).astype(tokens.dtype)
        contribution = (
            residual_gate[None, None, :] * memory * layer_active[:, None, None].astype(tokens.dtype)
        ).astype(tokens.dtype)
        enhanced = (tokens + contribution).astype(tokens.dtype)

        def state_norm(state: TactileTTTState) -> jax.Array:
            squared = sum(
                jnp.sum(jnp.square(value.astype(jnp.float32)), axis=tuple(range(1, value.ndim))) for value in state
            )
            return jnp.sqrt(squared)

        previous_state = (w1, b1, w2, b2)
        state_delta = tuple(new - old for new, old in zip(updated_state, previous_state, strict=True))
        contribution_norm = jnp.linalg.norm(contribution.astype(jnp.float32), axis=(1, 2))
        token_norm = jnp.linalg.norm(tokens.astype(jnp.float32), axis=(1, 2))
        stats = {
            "contact_gate": write_gate.astype(jnp.float32),
            "reconstruction": reconstruction * layer_active,
            "fast_weight_norm": state_norm(updated_state),
            "fast_weight_update_norm": state_norm(state_delta),
            "inner_lr": jnp.broadcast_to(inner_lr.astype(jnp.float32), write_gate.shape),
            "residual_gate_raw": jnp.broadcast_to(jnp.mean(residual_gate_raw), write_gate.shape),
            "residual_gate": jnp.broadcast_to(jnp.mean(jnp.tanh(residual_gate_raw)), write_gate.shape),
            "residual_gate_abs_mean": jnp.broadcast_to(
                jnp.mean(jnp.abs(jnp.tanh(residual_gate_raw))), write_gate.shape
            ),
            "memory_contribution_norm": contribution_norm,
            "memory_to_current_ratio": contribution_norm / (token_norm + 1e-6),
            "_layer_active": layer_active,
        }
        return enhanced, updated_state, stats

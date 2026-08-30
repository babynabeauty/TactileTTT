"""Small test-time-training memory for causal tactile histories."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp


class TactileTTTMemory(nnx.Module):
    """Associative fast-weight memory with a task-trained inner update.

    Slow parameters project tactile tokens into keys, values, and queries.  The
    per-episode fast matrix is explicit state and is updated once per action
    chunk.  Keeping the state outside the module makes reset/carry behavior the
    same during sequence training and deployment.
    """

    def __init__(
        self,
        *,
        token_dim: int,
        memory_dim: int,
        inner_lr: float,
        contact_threshold: float,
        contact_temperature: float,
        rngs: nnx.Rngs,
    ):
        self.token_dim = int(token_dim)
        self.memory_dim = int(memory_dim)
        self.inner_lr = float(inner_lr)
        self.contact_threshold = float(contact_threshold)
        self.contact_temperature = float(contact_temperature)

        self.token_norm = nnx.LayerNorm(num_features=self.token_dim, rngs=rngs)
        self.query_proj = nnx.Linear(self.token_dim, self.memory_dim, use_bias=False, rngs=rngs)
        self.key_proj = nnx.Linear(self.token_dim, self.memory_dim, use_bias=False, rngs=rngs)
        self.value_proj = nnx.Linear(self.token_dim, self.memory_dim, use_bias=False, rngs=rngs)
        self.output_proj = nnx.Linear(self.memory_dim, self.token_dim, use_bias=False, rngs=rngs)

        # A zero initial memory and zero residual gate preserve the pretrained
        # policy at initialization.  The outer action loss learns both.
        self.fast_weight_init = nnx.Param(jnp.zeros((self.memory_dim, self.memory_dim), dtype=jnp.float32))
        self.residual_gate = nnx.Param(jnp.asarray(0.0, dtype=jnp.float32))

    def initial_state(self, batch_size: int, *, dtype: jnp.dtype = jnp.float32) -> jax.Array:
        weight = jnp.asarray(self.fast_weight_init.value, dtype=dtype)
        return jnp.broadcast_to(weight[None, :, :], (batch_size, self.memory_dim, self.memory_dim))

    def contact_gate(self, contact_force: jax.Array) -> jax.Array:
        """Return one soft write gate per batch item.

        Args:
            contact_force: Unnormalized calc_force in Newtons, ``[B, T, F, 3]``.
        """
        if contact_force.ndim != 4 or contact_force.shape[-1] != 3:
            raise ValueError(f"Expected contact_force [B,T,F,3], got {contact_force.shape}.")
        magnitude = jnp.linalg.norm(contact_force.astype(jnp.float32), axis=-1)
        score = jnp.max(magnitude, axis=(1, 2))
        temperature = jnp.asarray(max(self.contact_temperature, 1e-6), dtype=jnp.float32)
        return jax.nn.sigmoid((score - self.contact_threshold) / temperature)

    def update(
        self,
        fast_weight: jax.Array,
        write_tokens: jax.Array,
        write_gate: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Apply one differentiable inner SGD update to each episode memory."""
        normalized = self.token_norm(write_tokens)
        keys = self.key_proj(normalized).astype(jnp.float32)
        values = self.value_proj(normalized).astype(jnp.float32)
        weight = fast_weight.astype(jnp.float32)

        prediction = jnp.einsum("bnd,bdm->bnm", keys, weight)
        error = prediction - values
        denom = float(write_tokens.shape[1] * self.memory_dim)
        gradient = 2.0 * jnp.einsum("bnd,bnm->bdm", keys, error) / denom
        effective_lr = self.inner_lr * write_gate.astype(jnp.float32)
        updated_weight = weight - effective_lr[:, None, None] * gradient
        reconstruction_loss = jnp.mean(jnp.square(error), axis=(1, 2))
        return updated_weight.astype(fast_weight.dtype), reconstruction_loss

    def read(self, fast_weight: jax.Array, current_tokens: jax.Array) -> jax.Array:
        """Read five state-free tactile queries and return enhanced tokens."""
        normalized = self.token_norm(current_tokens)
        queries = self.query_proj(normalized).astype(jnp.float32)
        memory = jnp.einsum("bnd,bdm->bnm", queries, fast_weight.astype(jnp.float32))
        memory = self.output_proj(memory.astype(current_tokens.dtype))
        gate = jnp.tanh(jnp.asarray(self.residual_gate.value, dtype=current_tokens.dtype))
        return current_tokens + gate * memory

    def step(
        self,
        fast_weight: jax.Array,
        write_tokens: jax.Array,
        current_tokens: jax.Array,
        contact_force: jax.Array,
        *,
        active: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
        write_gate = self.contact_gate(contact_force)
        if active is not None:
            write_gate = write_gate * active.astype(write_gate.dtype)
        updated_weight, reconstruction_loss = self.update(fast_weight, write_tokens, write_gate)
        enhanced_tokens = self.read(updated_weight, current_tokens)
        residual_gate_raw = jnp.asarray(self.residual_gate.value, dtype=jnp.float32)
        residual_gate = jnp.tanh(residual_gate_raw)
        memory_contribution = enhanced_tokens.astype(jnp.float32) - current_tokens.astype(jnp.float32)
        memory_contribution_norm = jnp.linalg.norm(memory_contribution, axis=(1, 2))
        current_token_norm = jnp.linalg.norm(current_tokens.astype(jnp.float32), axis=(1, 2))
        stats = {
            "contact_gate": write_gate,
            "reconstruction": reconstruction_loss,
            "fast_weight_norm": jnp.linalg.norm(updated_weight.astype(jnp.float32), axis=(1, 2)),
            "fast_weight_update_norm": jnp.linalg.norm(
                updated_weight.astype(jnp.float32) - fast_weight.astype(jnp.float32), axis=(1, 2)
            ),
            "residual_gate_raw": jnp.broadcast_to(residual_gate_raw, write_gate.shape),
            "residual_gate": jnp.broadcast_to(residual_gate, write_gate.shape),
            "memory_contribution_norm": memory_contribution_norm,
            "memory_to_current_ratio": memory_contribution_norm / (current_token_norm + 1e-6),
        }
        return enhanced_tokens, updated_weight, stats

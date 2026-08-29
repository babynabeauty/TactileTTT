import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np

from openpi.models.tactile_ttt import TactileTTTMemory


def _memory() -> TactileTTTMemory:
    return TactileTTTMemory(
        token_dim=32,
        memory_dim=8,
        inner_lr=0.1,
        contact_top_k=4,
        contact_threshold=1.0,
        contact_temperature=0.5,
        rngs=nnx.Rngs(0),
    )


def test_tactile_ttt_step_shapes_and_state_carry():
    memory = _memory()
    state = memory.initial_state(2)
    write_tokens = jnp.arange(2 * 20 * 32, dtype=jnp.float32).reshape(2, 20, 32) / 100.0
    current_tokens = jnp.arange(2 * 5 * 32, dtype=jnp.float32).reshape(2, 5, 32) / 100.0
    raw_tactile = jnp.ones((2, 16, 5, 120, 3), dtype=jnp.float32) * 3.0

    enhanced, updated, stats = memory.step(state, write_tokens, current_tokens, raw_tactile)

    assert enhanced.shape == (2, 5, 32)
    assert updated.shape == (2, 8, 8)
    assert stats["contact_gate"].shape == (2,)
    assert not np.allclose(updated, state)


def test_inactive_sequence_step_does_not_write_memory():
    memory = _memory()
    state = memory.initial_state(2)
    write_tokens = jnp.arange(2 * 20 * 32, dtype=jnp.float32).reshape(2, 20, 32) / 100.0
    current_tokens = jnp.arange(2 * 5 * 32, dtype=jnp.float32).reshape(2, 5, 32) / 100.0
    raw_tactile = jnp.ones((2, 16, 5, 120, 3), dtype=jnp.float32) * 3.0

    _, updated, _ = memory.step(
        state,
        write_tokens,
        current_tokens,
        raw_tactile,
        active=jnp.zeros((2,), dtype=jnp.float32),
    )

    assert np.allclose(updated, state)


def test_contact_gate_separates_no_contact_and_contact():
    memory = _memory()
    no_contact = jnp.zeros((1, 16, 5, 120, 3), dtype=jnp.float32)
    contact = no_contact.at[:, 7, 1, :8, 0].set(4.0)

    no_contact_gate = memory.contact_gate(no_contact)
    contact_gate = memory.contact_gate(contact)

    assert float(contact_gate[0]) > float(no_contact_gate[0])

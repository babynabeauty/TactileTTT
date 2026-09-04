import dataclasses

import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import gemma
from openpi.models.tactile_ttt import LayerwiseTactileTTT
from openpi.models.tactile_ttt import tactile_contact_gate


def _module() -> LayerwiseTactileTTT:
    return LayerwiseTactileTTT(
        width=32,
        memory_dim=8,
        mlp_dim=12,
        base_inner_lr=0.1,
        residual_gate_init=0.001,
    )


def _state(batch_size: int):
    key1, key2 = jax.random.split(jax.random.key(1))
    return (
        0.1 * jax.random.normal(key1, (batch_size, 8, 12)),
        jnp.zeros((batch_size, 12), dtype=jnp.float32),
        0.1 * jax.random.normal(key2, (batch_size, 12, 8)),
        jnp.zeros((batch_size, 8), dtype=jnp.float32),
    )


def test_layerwise_tactile_ttt_shapes_and_state_carry():
    module = _module()
    state = _state(2)
    tokens = jnp.arange(2 * 21 * 32, dtype=jnp.float32).reshape(2, 21, 32) / 100.0
    write_gate = jnp.array([0.8, 0.6], dtype=jnp.float32)
    variables = module.init(jax.random.key(0), tokens, state, write_gate, jnp.ones((2,)))

    enhanced, updated, stats = module.apply(variables, tokens, state, write_gate, jnp.ones((2,), dtype=jnp.float32))

    assert enhanced.shape == tokens.shape
    assert [value.shape for value in updated] == [(2, 8, 12), (2, 12), (2, 12, 8), (2, 8)]
    assert stats["contact_gate"].shape == (2,)
    assert stats["residual_gate"].shape == (2,)
    assert any(not np.allclose(new, old) for new, old in zip(updated, state, strict=True))


def test_zero_update_mask_does_not_write_fast_weights():
    module = _module()
    state = _state(2)
    tokens = jnp.arange(2 * 21 * 32, dtype=jnp.float32).reshape(2, 21, 32) / 100.0
    write_gate = jnp.ones((2,), dtype=jnp.float32)
    variables = module.init(jax.random.key(0), tokens, state, write_gate, jnp.ones((2,)))

    _, updated, stats = module.apply(variables, tokens, state, write_gate, jnp.zeros((2,), dtype=jnp.float32))

    assert all(np.allclose(new, old) for new, old in zip(updated, state, strict=True))
    assert np.allclose(stats["fast_weight_update_norm"], 0.0)


def test_v2_temporal_binding_uses_75_tactile_pairs_and_21_queries():
    module = _module()
    state = _state(2)
    queries = jnp.ones((2, 21, 32), dtype=jnp.float32)
    write_tokens = jax.random.normal(jax.random.key(2), (2, 75, 32))
    target_tokens = jax.random.normal(jax.random.key(3), (2, 75, 32))
    gate = jnp.ones((2,), dtype=jnp.float32)
    variables = module.init(
        jax.random.key(0),
        queries,
        state,
        gate,
        gate,
        write_tokens=write_tokens,
        target_tokens=target_tokens,
    )

    enhanced, updated, stats = module.apply(
        variables,
        queries,
        state,
        gate,
        gate,
        write_tokens=write_tokens,
        target_tokens=target_tokens,
    )

    assert enhanced.shape == queries.shape
    assert stats["reconstruction"].shape == (2,)
    assert any(not np.allclose(new, old) for new, old in zip(updated, state, strict=True))


def test_inactive_v2_layer_neither_reads_nor_writes():
    module = _module()
    state = _state(1)
    queries = jnp.ones((1, 21, 32), dtype=jnp.float32)
    write_tokens = jax.random.normal(jax.random.key(2), (1, 75, 32))
    target_tokens = jax.random.normal(jax.random.key(3), (1, 75, 32))
    gate = jnp.ones((1,), dtype=jnp.float32)
    variables = module.init(
        jax.random.key(0),
        queries,
        state,
        gate,
        gate,
        write_tokens=write_tokens,
        target_tokens=target_tokens,
        layer_active=0.0,
    )

    enhanced, updated, stats = module.apply(
        variables,
        queries,
        state,
        gate,
        gate,
        write_tokens=write_tokens,
        target_tokens=target_tokens,
        layer_active=0.0,
    )

    assert np.allclose(enhanced, queries)
    assert all(np.allclose(new, old) for new, old in zip(updated, state, strict=True))
    assert np.allclose(stats["_layer_active"], 0.0)


def test_vector_residual_gate_starts_near_point_zero_zero_one():
    module = _module()
    state = _state(1)
    tokens = jnp.ones((1, 21, 32), dtype=jnp.float32)
    write_gate = jnp.ones((1,), dtype=jnp.float32)
    variables = module.init(jax.random.key(0), tokens, state, write_gate, jnp.ones((1,)))
    _, _, stats = module.apply(variables, tokens, state, write_gate, jnp.ones((1,)))

    assert np.allclose(stats["residual_gate"], np.tanh(0.001), atol=1e-7)


def test_layerwise_tactile_ttt_preserves_bfloat16_transformer_carry_dtype():
    module = _module()
    state = _state(2)
    tokens = jnp.ones((2, 21, 32), dtype=jnp.bfloat16)
    write_gate = jnp.ones((2,), dtype=jnp.float32)
    variables = module.init(jax.random.key(0), tokens, state, write_gate, jnp.ones((2,)))

    enhanced, _, _ = module.apply(variables, tokens, state, write_gate, jnp.ones((2,)))

    assert enhanced.dtype == tokens.dtype


def test_contact_gate_separates_no_contact_and_contact():
    no_contact = jnp.zeros((1, 16, 5, 3), dtype=jnp.float32)
    at_threshold = no_contact.at[:, 7, 1, 0].set(4.3)
    contact = no_contact.at[:, 7, 1, 0].set(5.3)

    no_contact_gate = tactile_contact_gate(no_contact, threshold=4.3, temperature=0.5)
    threshold_gate = tactile_contact_gate(at_threshold, threshold=4.3, temperature=0.5)
    contact_gate = tactile_contact_gate(contact, threshold=4.3, temperature=0.5)

    assert np.allclose(threshold_gate, 0.5)
    assert float(no_contact_gate[0]) < 0.001
    assert float(contact_gate[0]) > 0.8


def test_contact_gate_uses_maximum_force_across_history_and_fingers():
    contact_force = jnp.zeros((2, 16, 5, 3), dtype=jnp.float32)
    contact_force = contact_force.at[0, 0, 0].set(jnp.array([3.0, 4.0, 0.0]))
    contact_force = contact_force.at[1, -1, -1].set(jnp.array([0.0, 0.0, 4.3]))

    gates = tactile_contact_gate(contact_force, threshold=4.3, temperature=0.5)

    assert np.allclose(gates[0], 1.0 / (1.0 + np.exp(-1.4)), atol=1e-6)
    assert np.allclose(gates[1], 0.5, atol=1e-6)


def test_gemma_action_expert_carries_one_fast_state_per_transformer_layer():
    base_config = gemma.get_config("dummy")
    student_config = dataclasses.replace(
        base_config,
        tactile_ttt_memory_dim=8,
        tactile_ttt_mlp_dim=12,
    )
    llm = nnx_bridge.ToNNX(gemma.Module(configs=[base_config, student_config], embed_dtype="bfloat16", adarms=False))
    llm.lazy_init(rngs=nnx.Rngs(0), method="init", use_adarms=[False, False])

    batch_size = 2
    depth = student_config.depth
    state = tuple(jnp.broadcast_to(value[None, ...], (depth, *value.shape)) for value in _state(batch_size))
    prefix = jnp.ones((batch_size, 3, base_config.width), dtype=jnp.bfloat16)
    student = jnp.ones((batch_size, 5, student_config.width), dtype=jnp.bfloat16)
    positions = jnp.broadcast_to(jnp.arange(8, dtype=jnp.int32), (batch_size, 8))
    mask = jnp.ones((batch_size, 8, 8), dtype=jnp.bool_)

    outputs, _, updated, stats = llm(
        [prefix, student],
        positions,
        mask,
        tactile_ttt_state=state,
        tactile_ttt_write_gate=jnp.ones((batch_size,), dtype=jnp.float32),
    )

    assert outputs[1].shape == student.shape
    assert outputs[1].dtype == student.dtype
    assert [value.shape[0] for value in updated] == [depth] * 4
    assert stats["reconstruction"].shape == (depth, batch_size)
    assert not np.allclose(updated[0][0], updated[0][1])

    # Prefix-only cache construction must keep the original two-result API.
    prefix_outputs, _ = llm(
        [prefix, None],
        jnp.broadcast_to(jnp.arange(3, dtype=jnp.int32), (batch_size, 3)),
        jnp.ones((batch_size, 3, 3), dtype=jnp.bool_),
    )
    assert prefix_outputs[0].shape == prefix.shape
    assert prefix_outputs[1] is None


def test_gemma_v2_updates_only_interleaved_layers():
    base_config = gemma.get_config("dummy")
    student_config = dataclasses.replace(
        base_config,
        tactile_ttt_memory_dim=8,
        tactile_ttt_mlp_dim=12,
        tactile_ttt_mode="v2",
        tactile_ttt_layer_period=2,
        tactile_ttt_layer_offset=1,
    )
    llm = nnx_bridge.ToNNX(gemma.Module(configs=[base_config, student_config], embed_dtype="bfloat16", adarms=False))
    llm.lazy_init(rngs=nnx.Rngs(0), method="init", use_adarms=[False, False])

    batch_size = 1
    state = tuple(
        jnp.broadcast_to(value[None, ...], (student_config.depth, *value.shape))
        for value in _state(batch_size)
    )
    prefix = jnp.ones((batch_size, 3, base_config.width), dtype=jnp.bfloat16)
    student = jnp.ones((batch_size, 21, student_config.width), dtype=jnp.bfloat16)
    positions = jnp.broadcast_to(jnp.arange(24, dtype=jnp.int32), (batch_size, 24))
    mask = jnp.ones((batch_size, 24, 24), dtype=jnp.bool_)
    write_tokens = jax.random.normal(
        jax.random.key(2), (batch_size, 75, student_config.width), dtype=jnp.bfloat16
    )
    target_tokens = jax.random.normal(
        jax.random.key(3), (batch_size, 75, student_config.width), dtype=jnp.bfloat16
    )

    _, _, updated, stats = llm(
        [prefix, student],
        positions,
        mask,
        tactile_ttt_state=state,
        tactile_ttt_write_gate=jnp.ones((batch_size,), dtype=jnp.float32),
        tactile_ttt_write_tokens=write_tokens,
        tactile_ttt_target_tokens=target_tokens,
    )

    for layer in range(student_config.depth):
        changed = not np.allclose(updated[0][layer], state[0][layer])
        assert changed == (layer % 2 == 1)
    assert np.allclose(stats["_layer_active"][:, 0], np.array([0.0, 1.0, 0.0, 1.0]))

    prefix_outputs, _ = llm(
        [prefix, None],
        jnp.broadcast_to(jnp.arange(3, dtype=jnp.int32), (batch_size, 3)),
        jnp.ones((batch_size, 3, 3), dtype=jnp.bool_),
    )
    assert prefix_outputs[0].shape == prefix.shape

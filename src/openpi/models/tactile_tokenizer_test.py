import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np

from openpi.models.tactile_tokenizer import DexterousForceTokenizer
from openpi.models.tactile_tokenizer import RawTactileSpatialTokenizer
from openpi.shared.normalize import NormStats
from openpi.transforms import Normalize


def _tokenizer() -> DexterousForceTokenizer:
    return DexterousForceTokenizer(
        output_dim=32,
        hidden_dim=16,
        num_fingers=5,
        dim_per_finger=3,
        future_segments=8,
        future_steps_per_segment=4,
        rngs=nnx.Rngs(0),
    )


def test_structured_force_token_counts():
    tokenizer = _tokenizer()
    history = jnp.zeros((2, 10, 5, 3))
    future = jnp.zeros((2, 32, 5, 3))

    history_tokens = tokenizer.encode_history(history, jnp.arange(-18, 2, 2) / 15.0)
    future_tokens = tokenizer.encode_future(future, jnp.arange(1, 33) / 15.0)

    assert history_tokens.shape == (2, 50, 32)
    assert future_tokens.shape == (2, 40, 32)


def test_future_force_can_pool_all_steps_per_finger():
    tokenizer = DexterousForceTokenizer(
        output_dim=32,
        hidden_dim=16,
        num_fingers=5,
        dim_per_finger=3,
        future_segments=1,
        future_steps_per_segment=32,
        rngs=nnx.Rngs(0),
    )
    future = jnp.zeros((2, 32, 5, 3))

    future_tokens = tokenizer.encode_future(future, jnp.arange(1, 33) / 15.0)

    assert future_tokens.shape == (2, 5, 32)


def test_finger_force_changes_only_corresponding_input_tokens():
    tokenizer = _tokenizer()
    history = jnp.zeros((1, 10, 5, 3))
    changed = history.at[:, :, 2, 0].set(1.0)
    times = jnp.arange(-18, 2, 2) / 15.0

    difference = jnp.mean(
        jnp.abs(tokenizer.encode_history(changed, times) - tokenizer.encode_history(history, times)),
        axis=-1,
    ).reshape(1, 10, 5)

    assert jnp.all(difference[:, :, 2] > 0)
    assert jnp.allclose(difference[:, :, [0, 1, 3, 4]], 0)


def test_structured_effort_normalization_preserves_per_finger_stats():
    values = np.arange(15, dtype=np.float32).reshape(1, 5, 3)
    stats = NormStats(
        mean=np.arange(15, dtype=np.float32),
        std=np.ones(15, dtype=np.float32),
        q01=None,
        q99=None,
    )

    normalized = Normalize({"effort": stats})({"effort": values})["effort"]

    assert normalized.shape == (1, 5, 3)
    assert np.allclose(normalized, 0)


def test_raw_tactile_can_pool_causal_sixteen_frames_to_twenty_tokens():
    tokenizer = RawTactileSpatialTokenizer(
        output_dim=32,
        hidden_dim=16,
        num_fingers=5,
        num_points=8,
        dim_per_point=3,
        future_segments=4,
        future_steps_per_segment=4,
        contact_top_k=4,
        contact_threshold=1.0,
        contact_temperature=0.5,
        rngs=nnx.Rngs(0),
    )
    tactile = jnp.zeros((2, 16, 5, 8, 3), dtype=jnp.float32)

    tokens = tokenizer.encode_temporal_segments(
        tactile,
        jnp.arange(-15, 1) / 15.0,
        num_segments=4,
        future=False,
    )

    assert tokens.shape == (2, 20, 32)

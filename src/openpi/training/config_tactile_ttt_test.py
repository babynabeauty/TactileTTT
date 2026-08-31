import flax.nnx as nnx
import jax.numpy as jnp

from openpi.training import config as training_config


def _representative_v1_state() -> nnx.State:
    return nnx.State(
        {
            "PaliGemma": {
                "img": {"kernel": nnx.Param(jnp.ones((1,)))},
                "llm": {
                    "layers": {
                        "attn_1": {"kernel": nnx.Param(jnp.ones((1,)))},
                        "mlp_1": {"kernel": nnx.Param(jnp.ones((1,)))},
                        "tactile_ttt": {
                            "key_proj": {"kernel": nnx.Param(jnp.ones((1,)))},
                            "inner_lr_raw": nnx.Param(jnp.ones((1,))),
                            "residual_gate": nnx.Param(jnp.ones((1,))),
                        },
                    }
                },
            },
            "student_force_tokenizer": {"kernel": nnx.Param(jnp.ones((1,)))},
            "action_out_proj_student": {"kernel": nnx.Param(jnp.ones((1,)))},
            "tactile_ttt_w1_init": nnx.Param(jnp.ones((1,))),
            "tactile_ttt_b1_init": nnx.Param(jnp.ones((1,))),
            "tactile_ttt_w2_init": nnx.Param(jnp.ones((1,))),
            "tactile_ttt_b2_init": nnx.Param(jnp.ones((1,))),
        }
    )


def _paths(state: nnx.State) -> set[str]:
    return {"/".join(map(str, path)) for path in state.flat_state()}


def test_tactile_ttt_warmup_trains_only_ttt_parameters():
    state = _representative_v1_state()
    config = training_config.get_config("pi05_tactile_ttt_v1_warmup")

    trainable_paths = _paths(state.filter(config.trainable_filter))
    frozen_paths = _paths(state.filter(config.freeze_filter))

    assert trainable_paths
    assert all("tactile_ttt" in path for path in trainable_paths)
    assert "PaliGemma/llm/layers/tactile_ttt/inner_lr_raw" in trainable_paths
    assert "PaliGemma/llm/layers/tactile_ttt/residual_gate" in trainable_paths
    assert "tactile_ttt_w1_init" in trainable_paths
    assert "PaliGemma/llm/layers/attn_1/kernel" in frozen_paths
    assert "PaliGemma/llm/layers/mlp_1/kernel" in frozen_paths
    assert "PaliGemma/img/kernel" in frozen_paths
    assert "student_force_tokenizer/kernel" in frozen_paths
    assert "action_out_proj_student/kernel" in frozen_paths


def test_joint_v1_keeps_the_same_model_but_removes_warmup_freeze():
    warmup = training_config.get_config("pi05_tactile_ttt_v1_warmup")
    joint = training_config.get_config("pi05_tactile_ttt_v1")
    state = _representative_v1_state()

    assert warmup.model == joint.model
    assert warmup.data == joint.data
    assert _paths(state.filter(joint.trainable_filter)) == _paths(state)
    assert not _paths(state.filter(joint.freeze_filter))
    assert joint.model.disable_future_tactile
    assert joint.model.tactile_ttt_enabled

import numpy as np

from openpi.training import weight_loaders


def test_augment_with_latent_flow_head_weights_maps_pi05_heads():
    loaded = {
        "action_in_proj": {
            "kernel": np.arange(6, dtype=np.float32).reshape(2, 3),
            "bias": np.arange(3, dtype=np.float32),
        },
        "action_out_proj": {
            "kernel": np.arange(6, dtype=np.float32).reshape(3, 2),
            "bias": np.arange(2, dtype=np.float32),
        },
        "time_mlp_in": {"kernel": np.eye(3, dtype=np.float32)},
        "time_mlp_out": {"kernel": np.eye(3, dtype=np.float32) * 2},
    }
    reference = {
        "action_in_proj_student": {
            "kernel": np.zeros((2, 3), dtype=np.float32),
            "bias": np.zeros((3,), dtype=np.float32),
        },
        "action_out_proj_student": {
            "kernel": np.zeros((3, 2), dtype=np.float32),
            "bias": np.zeros((2,), dtype=np.float32),
        },
        "student_time_mlp_in": {"kernel": np.zeros((3, 3), dtype=np.float32)},
        "student_time_mlp_out": {"kernel": np.zeros((3, 3), dtype=np.float32)},
        "action_in_proj_teacher": {
            "kernel": np.zeros((2, 3), dtype=np.float32),
            "bias": np.zeros((3,), dtype=np.float32),
        },
        "action_out_proj_teacher": {
            "kernel": np.zeros((3, 2), dtype=np.float32),
            "bias": np.zeros((2,), dtype=np.float32),
        },
        "teacher_time_mlp_in": {"kernel": np.zeros((3, 3), dtype=np.float32)},
        "teacher_time_mlp_out": {"kernel": np.zeros((3, 3), dtype=np.float32)},
    }

    augmented = weight_loaders._augment_with_latent_flow_head_weights(loaded, reference)

    for source, targets in {
        "action_in_proj": ("action_in_proj_student", "action_in_proj_teacher"),
        "action_out_proj": ("action_out_proj_student", "action_out_proj_teacher"),
        "time_mlp_in": ("student_time_mlp_in", "teacher_time_mlp_in"),
        "time_mlp_out": ("student_time_mlp_out", "teacher_time_mlp_out"),
    }.items():
        for target in targets:
            for leaf, expected in loaded[source].items():
                assert np.array_equal(augmented[target][leaf], expected)


def test_augment_with_latent_flow_head_weights_skips_shape_mismatch():
    loaded = {"action_in_proj": {"kernel": np.ones((2, 3), dtype=np.float32)}}
    reference = {"action_in_proj_student": {"kernel": np.zeros((4, 3), dtype=np.float32)}}

    augmented = weight_loaders._augment_with_latent_flow_head_weights(loaded, reference)

    assert "action_in_proj_student" not in augmented

import numpy as np

from openpi.models.model import ModelType
from openpi.policies import xhand_policy


def test_extract_calc_force_history_converts_lsb_to_newtons_and_preserves_sensor_order():
    num_frames = 16
    state_dim = (
        xhand_policy.TACTILE_BLOCK_START
        + xhand_policy.TACTILE_SENSOR_COUNT * xhand_policy.TACTILE_BLOCK_SIZE
    )
    state_seq = np.zeros((num_frames, state_dim), dtype=np.float32)
    expected = np.zeros((num_frames, xhand_policy.TACTILE_SENSOR_COUNT, 3), dtype=np.float32)

    for frame in range(num_frames):
        for finger in range(xhand_policy.TACTILE_SENSOR_COUNT):
            force_lsb = np.array([frame, finger, 10 * frame + finger], dtype=np.float32)
            start = (
                xhand_policy.TACTILE_BLOCK_START
                + finger * xhand_policy.TACTILE_BLOCK_SIZE
                + xhand_policy.TACTILE_CALC_FORCE_OFFSET
            )
            state_seq[frame, start : start + 3] = force_lsb
            expected[frame, finger] = force_lsb * xhand_policy.CALC_FORCE_LSB_TO_NEWTON

    extracted = xhand_policy.XHandTactileFlowInputs(
        model_type=ModelType.PI05,
        include_tactile_contact_force=True,
        tactile_history_frames=num_frames,
    )._extract_calc_force(state_seq)  # noqa: SLF001

    assert extracted.shape == (16, 5, 3)
    assert extracted.dtype == np.float32
    np.testing.assert_allclose(extracted, expected, rtol=0.0, atol=1e-6)

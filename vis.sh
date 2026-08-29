
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/env/.venv/bin/python}"

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" scripts/infer_future_query_probe.py \
  --repo-id llly/all_0409_stage_flow \
  --probe-checkpoint-dir checkpoints/future_query_probe/pi0_latent_flow_noise/probe_from_30k_2gpu_/ \
  --batch-size 8 \
  --max-batches 16 \
  --output-dir outputs/future_probe_eval



CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" scripts/visualize_future_query_probe_episode.py \
  --episode 0 \
  --predict-stride 32 \
  --output-dir outputs/future_probe_episode_vis \
  --highres-arrow-min-magnitude 6.0 \
  --highres-arrow-thickness 2 \
  --highres-arrow-render-scale 1 \
  --force-plot-font-size 18 \
  --force-plot-legend-font-size 14 \
  --force-plot-tick-font-size 16

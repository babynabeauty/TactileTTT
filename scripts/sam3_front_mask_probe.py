"""Run SAM3 image inference on one front-camera LeRobot frame.

This is a small bridge script for validating that the local SAM3 source and
local checkpoints work inside the forceWAM environment.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch
from PIL import Image, ImageDraw
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from torchcodec.decoders import VideoDecoder


def read_frame(video_path: pathlib.Path, frame_index: int) -> Image.Image:
    decoder = VideoDecoder(video_path, dimension_order="NHWC", device="cpu")
    frame = decoder.get_frame_at(frame_index).data.numpy()
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return Image.fromarray(frame)


def overlay_masks(image: Image.Image, masks: torch.Tensor, boxes: torch.Tensor, scores: torch.Tensor) -> Image.Image:
    out = image.convert("RGBA")
    if masks.numel() == 0:
        return out.convert("RGB")

    colors = [
        (245, 191, 35, 115),
        (0, 210, 210, 130),
        (220, 80, 220, 110),
        (80, 200, 80, 110),
        (255, 120, 40, 110),
    ]
    masks_np = masks.detach().cpu().numpy()
    boxes_np = boxes.detach().float().cpu().numpy()
    scores_np = scores.detach().float().cpu().numpy()
    draw = ImageDraw.Draw(out)

    for idx, mask in enumerate(masks_np):
        color = colors[idx % len(colors)]
        if mask.ndim == 3:
            mask = mask[0]
        alpha = (mask.astype(np.uint8) * color[3])
        overlay = Image.new("RGBA", image.size, color[:3] + (0,))
        overlay.putalpha(Image.fromarray(alpha))
        out = Image.alpha_composite(out, overlay)
        draw = ImageDraw.Draw(out)
        x0, y0, x1, y1 = boxes_np[idx].tolist()
        draw.rectangle((x0, y0, x1, y1), outline=color[:3] + (255,), width=3)
        draw.text((x0, max(0, y0 - 14)), f"{idx}:{scores_np[idx]:.2f}", fill=color[:3] + (255,))
    return out.convert("RGB")


def save_prompt_outputs(output_dir: pathlib.Path, image: Image.Image, name: str, output: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    masks = output.get("masks", torch.empty(0))
    boxes = output.get("boxes", torch.empty(0, 4))
    scores = output.get("scores", torch.empty(0))
    overlay_masks(image, masks, boxes, scores).save(output_dir / f"{name}_overlay.png")
    torch.save(
        {
            "masks": masks.detach().cpu(),
            "boxes": boxes.detach().float().cpu(),
            "scores": scores.detach().float().cpu(),
        },
        output_dir / f"{name}_raw.pt",
    )
    print(f"{name}: masks={tuple(masks.shape)} boxes={tuple(boxes.shape)} scores={scores.detach().float().cpu().tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        type=pathlib.Path,
        default=pathlib.Path("/data/shared_workspace/zhangshiqi/dataset/tactile_xhand_ur7e/grasp_pipette_and_press_button"),
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=pathlib.Path("/data/shared_workspace/zhangshiqi/hf/SAM/sam3/sam3.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--robot-text", default="robot arm")
    parser.add_argument("--hand-text", default="robot hand")
    parser.add_argument(
        "--object-box",
        default="226,190,262,370",
        help="Object box in xyxy pixel coordinates. Defaults around the pipette/button object.",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("outputs/sam3_front_mask_probe"))
    args = parser.parse_args()

    ep = f"episode_{args.episode_index:06d}"
    video_path = args.repo_id / "videos" / "chunk-000" / "observation.images.cam_front" / f"{ep}.mp4"
    image = read_frame(video_path, args.frame_index)

    model = build_sam3_image_model(
        checkpoint_path=str(args.checkpoint),
        load_from_HF=False,
        device=args.device,
        eval_mode=True,
    )
    processor = Sam3Processor(model, device=args.device, confidence_threshold=args.confidence_threshold)
    amp_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "none": None,
    }[args.amp_dtype]

    def amp_context():
        if args.device.startswith("cuda") and amp_dtype is not None:
            return torch.autocast(device_type="cuda", dtype=amp_dtype)
        return torch.inference_mode()

    with amp_context():
        state = processor.set_image(image)

    out = args.output_dir / f"{ep}_frame_{args.frame_index:06d}"
    out.mkdir(parents=True, exist_ok=True)
    image.save(out / "input.png")

    with amp_context():
        robot_out = processor.set_text_prompt(args.robot_text, state=state)
    save_prompt_outputs(out, image, "robot_text", robot_out)

    processor.reset_all_prompts(state)
    with amp_context():
        state = processor.set_image(image, state={})
        hand_out = processor.set_text_prompt(args.hand_text, state=state)
    save_prompt_outputs(out, image, "hand_text", hand_out)

    processor.reset_all_prompts(state)
    with amp_context():
        state = processor.set_image(image, state={})
    x0, y0, x1, y1 = [float(x) for x in args.object_box.split(",")]
    width, height = image.size
    cx = ((x0 + x1) * 0.5) / width
    cy = ((y0 + y1) * 0.5) / height
    bw = (x1 - x0) / width
    bh = (y1 - y0) / height
    with amp_context():
        object_out = processor.add_geometric_prompt([cx, cy, bw, bh], True, state=state)
    save_prompt_outputs(out, image, "object_box", object_out)

    print(f"Saved SAM3 probe outputs to {out.resolve()}")


if __name__ == "__main__":
    main()

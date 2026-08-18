#!/usr/bin/env python3
"""
Standalone inference script for the I4C SEM image restoration model.

Example:
    python inference.py \
        --input_dir ./test_images \
        --output_dir ./outputs \
        --weights ./weights/I4C_SEM_SR_33dB.pth

The script accepts .npy grayscale arrays and common image files.
For .npy input, the output is saved as .npy with the same filename.
For image input, the output is saved as PNG with the same stem.

Expected model input:
    1 x H x W grayscale image, typically 128 x 128.

Expected model output:
    1 x 2H x 2W grayscale image.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from models.sem_sr import SEMSuperResolution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="I4C SEM image restoration inference"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        type=Path,
        help="Directory containing test images (.npy/.png/.jpg/.jpeg/.tif/.tiff).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Directory in which restored outputs will be written.",
    )
    parser.add_argument(
        "--weights",
        default="weights/I4C_SEM_SR_33dB.pth",
        type=Path,
        help="Path to trained PyTorch checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device. Default: auto.",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(model: torch.nn.Module, weights_path: Path, device: torch.device) -> dict:
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}\n"
            "Pass the correct path with --weights."
        )

    checkpoint = torch.load(
        weights_path,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata = {
            "psnr": checkpoint.get("psnr"),
            "ssim": checkpoint.get("ssim"),
            "epoch": checkpoint.get("epoch"),
        }
    else:
        state_dict = checkpoint
        metadata = {}

    # Support checkpoints saved from DataParallel.
    state_dict = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    model.load_state_dict(state_dict, strict=True)
    return metadata


def load_npy(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise ValueError(
            f"{path}: expected a 2-D grayscale array after squeeze, got shape {arr.shape}"
        )

    # Match the submitted preprocessing: clip to [0, 1].
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def load_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def save_png(arr: np.ndarray, path: Path) -> None:
    img = np.clip(arr * 255.0, 0.0, 255.0).round().astype(np.uint8)
    Image.fromarray(img, mode="L").save(path)


def iter_inputs(input_dir: Path) -> Iterable[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    supported = {".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    return sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in supported)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = SEMSuperResolution(features=64, blocks=12).to(device)
    metadata = load_checkpoint(model, args.weights, device)
    model.eval()

    inputs = list(iter_inputs(args.input_dir))
    if not inputs:
        raise RuntimeError(
            f"No supported input images found in {args.input_dir}"
        )

    print(f"Device: {device}")
    print(f"Inputs: {len(inputs)}")
    if metadata:
        print(f"Checkpoint PSNR: {metadata.get('psnr')}")
        print(f"Checkpoint SSIM: {metadata.get('ssim')}")
        print(f"Checkpoint epoch: {metadata.get('epoch')}")

    with torch.inference_mode():
        for index, path in enumerate(inputs, start=1):
            if path.suffix.lower() == ".npy":
                image = load_npy(path)
            else:
                image = load_image(path)

            tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
            tensor = tensor.to(device)

            restored = model(tensor)
            restored = torch.clamp(restored, 0.0, 1.0)
            restored = restored.squeeze(0).squeeze(0).cpu().numpy()

            if path.suffix.lower() == ".npy":
                np.save(
                    args.output_dir / path.name,
                    restored.astype(np.float32),
                )
            else:
                save_png(
                    restored,
                    args.output_dir / f"{path.stem}.png",
                )

            if index % 50 == 0 or index == len(inputs):
                print(f"Processed {index}/{len(inputs)}")

    print(f"Restored outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()

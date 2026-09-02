"""Validate an ONNX model by building it with TensorRT's trtexec."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import onnx


def validate_with_tensorrt(
    model_path: Path,
    *,
    min_shapes: str | None,
    opt_shapes: str | None,
    max_shapes: str | None,
) -> None:
    """Check ONNX validity, then parse and build the model with TensorRT."""
    model_path = model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    # This catches ONNX-specification errors before invoking TensorRT.
    model = onnx.load(model_path, load_external_data=True)
    onnx.checker.check_model(model)
    print(f"ONNX checker passed: {model_path}")

    trtexec = shutil.which("trtexec")
    if trtexec is None:
        raise RuntimeError(
            "trtexec was not found. Run this validator in an environment with "
            "TensorRT installed and trtexec available on PATH."
        )

    command = [
        trtexec,
        f"--onnx={model_path}",
        "--skipInference",
    ]

    shape_options = {
        "--minShapes": min_shapes,
        "--optShapes": opt_shapes,
        "--maxShapes": max_shapes,
    }
    supplied_shapes = [value is not None for value in shape_options.values()]
    if any(supplied_shapes) and not all(supplied_shapes):
        raise ValueError(
            "Dynamic shape validation requires --min-shapes, --opt-shapes, "
            "and --max-shapes together."
        )
    for option, value in shape_options.items():
        if value is not None:
            command.append(f"{option}={value}")

    print("Running:", " ".join(command))
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"TensorRT rejected the model or could not build an engine "
            f"(trtexec exit code {result.returncode})."
        )

    print("TensorRT parse and engine build passed.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate an ONNX model against ONNX and the installed TensorRT."
    )
    parser.add_argument("model", type=Path, help="Path to the ONNX model")
    parser.add_argument(
        "--min-shapes",
        help='Minimum dynamic input shapes, for example "x:1x4"',
    )
    parser.add_argument(
        "--opt-shapes",
        help='Optimal dynamic input shapes, for example "x:8x4"',
    )
    parser.add_argument(
        "--max-shapes",
        help='Maximum dynamic input shapes, for example "x:32x4"',
    )
    return parser.parse_args()


def main() -> None:
    """Run model validation."""
    args = parse_args()
    validate_with_tensorrt(
        args.model,
        min_shapes=args.min_shapes,
        opt_shapes=args.opt_shapes,
        max_shapes=args.max_shapes,
    )


if __name__ == "__main__":
    main()

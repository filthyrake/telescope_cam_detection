#!/usr/bin/env python3
"""
YOLOX to TensorRT Conversion Script
Converts YOLOX PyTorch models to TensorRT for 1.5-2.4x inference speedup.

Usage:
    python tools/convert_yolox_to_tensorrt.py \
        --model yolox-s \
        --weights models/yolox/yolox_s.pth \
        --input-size 640 640 \
        --output models/yolox/yolox_s_trt.pth

Performance Expectations (NVIDIA A30):
    YOLOX-S @ 640x640:   11-21ms  → 7-12ms   (1.5-1.7x speedup)
    YOLOX-S @ 1920x1920: 150-250ms → 90-150ms (1.5-1.7x speedup)

Issue: #157 (Convert YOLOX to TensorRT for 1.5-2.4x inference speedup)
"""

import argparse
import sys
import time
import logging
from pathlib import Path

import torch
import torch_tensorrt

# Add YOLOX to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.yolox_detector import YOLOXDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_parser():
    parser = argparse.ArgumentParser("YOLOX TensorRT Conversion")

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="yolox-s",
        help="YOLOX model variant (yolox-s, yolox-m, yolox-l, yolox-x)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="models/yolox/yolox_s.pth",
        help="Path to YOLOX PyTorch weights",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[640, 640],
        help="Input size (height width)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for TensorRT model (default: <weights>_trt.pth)",
    )

    # TensorRT configuration
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Enable FP16 precision (default: True)",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_false",
        dest="fp16",
        help="Disable FP16 precision (use FP32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for conversion",
    )
    parser.add_argument(
        "--workspace-size",
        type=int,
        default=4,
        help="TensorRT workspace size in GB (default: 4GB)",
    )

    # Benchmarking
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark PyTorch vs TensorRT performance",
    )
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations before benchmarking (default: 10)",
    )

    return parser


def load_yolox_model(model_name: str, weights_path: str, device: str, input_size: tuple):
    """
    Load YOLOX model from weights.

    Args:
        model_name: YOLOX model variant (e.g., "yolox-s")
        weights_path: Path to PyTorch weights
        device: Device to load model on
        input_size: Input size (height, width)

    Returns:
        YOLOXDetector instance with loaded model
    """
    logger.info(f"Loading YOLOX model: {model_name}")
    logger.info(f"Weights: {weights_path}")
    logger.info(f"Input size: {input_size}")

    # Create detector (will load model)
    detector = YOLOXDetector(
        model_name=model_name,
        model_path=weights_path,
        device=device,
        input_size=input_size,
        conf_threshold=0.25,
        nms_threshold=0.45,
        wildlife_only=False  # Don't filter during conversion
    )

    if not detector.load_model():
        raise RuntimeError(f"Failed to load model from {weights_path}")

    logger.info("✓ YOLOX model loaded successfully")
    return detector


def convert_to_tensorrt(
    detector: YOLOXDetector,
    input_size: tuple,
    fp16: bool = True,
    workspace_size_gb: int = 4,
):
    """
    Convert YOLOX model to TensorRT.

    Args:
        detector: YOLOXDetector instance with loaded model
        input_size: Input size (height, width)
        fp16: Enable FP16 precision
        workspace_size_gb: TensorRT workspace size in GB

    Returns:
        TensorRT compiled model
    """
    logger.info("Converting YOLOX to TensorRT...")
    logger.info(f"  FP16 mode: {fp16}")
    logger.info(f"  Workspace size: {workspace_size_gb}GB")

    model = detector.model
    model.eval()

    # Create example input
    example_input = torch.randn(1, 3, input_size[0], input_size[1]).to(detector.device)
    input_shape = list(example_input.shape)

    # Configure TensorRT compilation
    enabled_precisions = {torch.float32}
    if fp16:
        enabled_precisions.add(torch.half)

    # Parse device string (e.g., "cuda:0" -> torch_tensorrt.Device)
    device_str = str(detector.device)
    if device_str.startswith("cuda"):
        gpu_id = int(device_str.split(":")[-1]) if ":" in device_str else 0
        trt_device = torch_tensorrt.Device(f"cuda:{gpu_id}")
    else:
        trt_device = torch_tensorrt.Device("cuda:0")  # Default to cuda:0

    # Compile model with TensorRT
    logger.info("Compiling model with TensorRT (this may take several minutes)...")
    start_time = time.time()

    try:
        trt_model = torch_tensorrt.compile(
            model,
            inputs=[
                torch_tensorrt.Input(
                    min_shape=input_shape,
                    opt_shape=input_shape,
                    max_shape=input_shape,
                    dtype=torch.half if fp16 else torch.float32,
                )
            ],
            enabled_precisions=enabled_precisions,
            workspace_size=workspace_size_gb * (1 << 30),  # Convert GB to bytes
            truncate_long_and_double=True,
            device=trt_device,
        )
    except Exception as e:
        logger.error(f"TensorRT compilation failed: {e}")
        logger.error("This may be due to unsupported operations or insufficient GPU memory")
        raise

    compile_time = time.time() - start_time
    logger.info(f"✓ TensorRT compilation completed in {compile_time:.1f}s")

    return trt_model


def benchmark_models(
    pytorch_model: torch.nn.Module,
    trt_model: torch.nn.Module,
    input_size: tuple,
    device: str,
    iterations: int = 100,
    warmup: int = 10,
):
    """
    Benchmark PyTorch vs TensorRT performance.

    Args:
        pytorch_model: Original PyTorch model
        trt_model: TensorRT compiled model
        input_size: Input size (height, width)
        device: Device to benchmark on
        iterations: Number of benchmark iterations
        warmup: Warmup iterations
    """
    logger.info(f"\nBenchmarking PyTorch vs TensorRT ({iterations} iterations)...")

    # Create test input
    test_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)

    # Check if CUDA is available
    is_cuda = torch.cuda.is_available() and str(device).startswith('cuda')

    # Benchmark PyTorch model
    logger.info("Benchmarking PyTorch model...")
    pytorch_model.eval()

    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = pytorch_model(test_input)

        # Synchronize GPU (only if CUDA)
        if is_cuda:
            torch.cuda.synchronize()

        # Benchmark
        pytorch_times = []
        for _ in range(iterations):
            start = time.time()
            _ = pytorch_model(test_input)
            if is_cuda:
                torch.cuda.synchronize()
            pytorch_times.append((time.time() - start) * 1000)  # Convert to ms

    pytorch_avg = sum(pytorch_times) / len(pytorch_times)
    pytorch_std = (sum((t - pytorch_avg) ** 2 for t in pytorch_times) / len(pytorch_times)) ** 0.5

    # Benchmark TensorRT model
    logger.info("Benchmarking TensorRT model...")
    trt_model.eval()

    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = trt_model(test_input)

        # Synchronize GPU (only if CUDA)
        if is_cuda:
            torch.cuda.synchronize()

        # Benchmark
        trt_times = []
        for _ in range(iterations):
            start = time.time()
            _ = trt_model(test_input)
            if is_cuda:
                torch.cuda.synchronize()
            trt_times.append((time.time() - start) * 1000)  # Convert to ms

    trt_avg = sum(trt_times) / len(trt_times)
    trt_std = (sum((t - trt_avg) ** 2 for t in trt_times) / len(trt_times)) ** 0.5

    # Calculate speedup
    speedup = pytorch_avg / trt_avg

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 80)
    logger.info(f"PyTorch Model:")
    logger.info(f"  Average: {pytorch_avg:.2f}ms ± {pytorch_std:.2f}ms")
    logger.info(f"  Min:     {min(pytorch_times):.2f}ms")
    logger.info(f"  Max:     {max(pytorch_times):.2f}ms")
    logger.info("")
    logger.info(f"TensorRT Model:")
    logger.info(f"  Average: {trt_avg:.2f}ms ± {trt_std:.2f}ms")
    logger.info(f"  Min:     {min(trt_times):.2f}ms")
    logger.info(f"  Max:     {max(trt_times):.2f}ms")
    logger.info("")
    logger.info(f"Speedup: {speedup:.2f}x")
    logger.info(f"Latency Reduction: {pytorch_avg - trt_avg:.2f}ms ({(1 - trt_avg/pytorch_avg)*100:.1f}%)")
    logger.info("=" * 80)


def main():
    args = make_parser().parse_args()

    # Validate inputs
    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error(f"Weights file not found: {weights_path}")
        sys.exit(1)

    # Determine output path
    if args.output is None:
        output_path = weights_path.parent / f"{weights_path.stem}_trt.pth"
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("YOLOX TensorRT Conversion - Issue #157")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Input weights: {weights_path}")
    logger.info(f"Output TensorRT model: {output_path}")
    logger.info(f"Input size: {args.input_size[0]}x{args.input_size[1]}")
    logger.info(f"FP16: {args.fp16}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 80)

    # Step 1: Load YOLOX model
    detector = load_yolox_model(
        model_name=args.model,
        weights_path=str(weights_path),
        device=args.device,
        input_size=tuple(args.input_size),
    )

    # Keep reference to original model for benchmarking
    pytorch_model = detector.model

    # Step 2: Convert to TensorRT
    trt_model = convert_to_tensorrt(
        detector=detector,
        input_size=tuple(args.input_size),
        fp16=args.fp16,
        workspace_size_gb=args.workspace_size,
    )

    # Step 3: Save TensorRT model
    logger.info(f"Saving TensorRT model to: {output_path}")
    torch.jit.save(trt_model, str(output_path))
    logger.info(f"✓ TensorRT model saved successfully")

    # Step 4: Benchmark (optional)
    if args.benchmark:
        benchmark_models(
            pytorch_model=pytorch_model,
            trt_model=trt_model,
            input_size=tuple(args.input_size),
            device=args.device,
            iterations=args.benchmark_iterations,
            warmup=args.warmup,
        )

    logger.info("\n" + "=" * 80)
    logger.info("CONVERSION COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"TensorRT model saved to: {output_path}")
    logger.info("")
    logger.info("To use the TensorRT model, update your config/config.yaml:")
    logger.info(f"  detection.model.weights: \"{output_path}\"")
    logger.info(f"  detection.model.use_tensorrt: true")
    logger.info("")
    logger.info("Expected speedup: 1.5-2.4x faster inference")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

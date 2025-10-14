"""
Image Enhancement Module
Pre-processes bounding box crops before Stage 2 species classification.

Option B: Real-ESRGAN + CLAHE + Bilateral Denoising
- Real-ESRGAN 4x upscaling for small/distant animals
- CLAHE for contrast enhancement
- Bilateral filtering for denoising while preserving edges

Dependencies:
    pip install realesrgan basicsr facexlib gfpgan

Note: basicsr may require torchvision compatibility patch for newer versions.
If you encounter "No module named 'torchvision.transforms.functional_tensor'",
patch the import in basicsr/data/degradations.py to use functional instead.
"""

import cv2
import torch
import numpy as np
import logging
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class ImageEnhancer:
    """
    Image enhancement pipeline for wildlife classification.

    Supports multiple enhancement methods:
    - "none": No enhancement
    - "clahe": CLAHE + bilateral denoising only (fast, ~15ms)
    - "realesrgan": Real-ESRGAN 4x upscaling + CLAHE + bilateral (slow, ~1s)
    """

    def __init__(
        self,
        method: str = "realesrgan",
        device: Optional[str] = None,
        realesrgan_model_path: Optional[str] = None,
        realesrgan_scale: int = 4,
        realesrgan_tile: int = 512,
        realesrgan_tile_pad: int = 10,
        realesrgan_half: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
        bilateral_d: int = 9,
        bilateral_sigma_color: int = 75,
        bilateral_sigma_space: int = 75
    ):
        """
        Initialize image enhancer.

        Args:
            method: Enhancement method ("none", "clahe", "realesrgan")
            device: Device for Real-ESRGAN inference (auto-detects if None)
            realesrgan_model_path: Path to Real-ESRGAN model weights
            realesrgan_scale: Upscaling factor (2, 3, or 4)
            realesrgan_tile: Tile size for processing (0 = no tiling)
            realesrgan_tile_pad: Padding for tiles
            realesrgan_half: Use FP16 for 2x speedup
            clahe_clip_limit: CLAHE clip limit (higher = more contrast)
            clahe_tile_grid_size: CLAHE tile grid size
            bilateral_d: Bilateral filter diameter
            bilateral_sigma_color: Bilateral filter color sigma
            bilateral_sigma_space: Bilateral filter space sigma
        """
        self.method = method
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Real-ESRGAN settings
        self.realesrgan_model_path = realesrgan_model_path
        self.realesrgan_scale = realesrgan_scale
        self.realesrgan_tile = realesrgan_tile
        self.realesrgan_tile_pad = realesrgan_tile_pad
        self.realesrgan_half = realesrgan_half
        self.realesrgan_model = None

        # CLAHE settings
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid_size
        )

        # Bilateral filter settings
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space

        logger.info(f"ImageEnhancer initialized: method={method}")

        # Load Real-ESRGAN model if needed
        if method == "realesrgan":
            if realesrgan_model_path is None:
                raise ValueError("realesrgan_model_path required when method='realesrgan'")
            self._load_realesrgan()

    def _load_realesrgan(self):
        """Load Real-ESRGAN model"""
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            logger.info(f"Loading Real-ESRGAN model: {self.realesrgan_model_path}")

            # Create model architecture (RRDBNet for Real-ESRGAN)
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4
            )

            # Create upsampler
            self.realesrgan_model = RealESRGANer(
                scale=self.realesrgan_scale,
                model_path=self.realesrgan_model_path,
                model=model,
                tile=self.realesrgan_tile,
                tile_pad=self.realesrgan_tile_pad,
                pre_pad=0,
                half=self.realesrgan_half,
                device=self.device
            )

            logger.info(f"✓ Real-ESRGAN loaded (scale={self.realesrgan_scale}x, tile={self.realesrgan_tile})")
            if self.realesrgan_half:
                logger.info("  FP16 mode enabled (2x faster)")

        except Exception as e:
            logger.error(f"Failed to load Real-ESRGAN: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def enhance_clahe_bilateral(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE + bilateral denoising.

        Fast enhancement method (~15ms) for improving contrast and reducing noise.
        Works on grayscale or color images.

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            Enhanced image (same format as input)
        """
        # Convert to grayscale if color (CLAHE works on single channel)
        if len(image.shape) == 3:
            # Color image - apply to luminance channel
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE to luminance
            l_clahe = self.clahe.apply(l)

            # Merge back
            lab_clahe = cv2.merge([l_clahe, a, b])
            enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale image
            enhanced = self.clahe.apply(image)

        # Apply bilateral filter (denoise while preserving edges)
        enhanced = cv2.bilateralFilter(
            enhanced,
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sigma_color,
            sigmaSpace=self.bilateral_sigma_space
        )

        return enhanced

    def enhance_realesrgan(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Real-ESRGAN 4x upscaling + CLAHE + bilateral.

        Slow enhancement method (~1s) for small/distant animals.
        Upscales image 4x, then applies contrast and denoising.

        Args:
            image: Input image (BGR)

        Returns:
            Enhanced upscaled image (4x size, BGR)
        """
        if self.realesrgan_model is None:
            raise RuntimeError("Real-ESRGAN model not loaded")

        # Upscale with Real-ESRGAN
        output, _ = self.realesrgan_model.enhance(image, outscale=self.realesrgan_scale)

        # Apply CLAHE + bilateral to upscaled image
        output = self.enhance_clahe_bilateral(output)

        return output

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image based on configured method.

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            Enhanced image
        """
        if self.method == "none":
            return image

        elif self.method == "clahe":
            return self.enhance_clahe_bilateral(image)

        elif self.method == "realesrgan":
            return self.enhance_realesrgan(image)

        else:
            logger.warning(f"Unknown enhancement method: {self.method}, returning original")
            return image


# Standalone test function
if __name__ == "__main__":
    import time

    # Test image (simulated small wildlife detection crop)
    test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

    print("Testing ImageEnhancer...")
    print("="*60)

    # Test Method 1: CLAHE only
    print("\n1. Testing CLAHE + Bilateral...")
    enhancer_clahe = ImageEnhancer(method="clahe")

    start = time.time()
    enhanced_clahe = enhancer_clahe.enhance(test_image)
    elapsed = (time.time() - start) * 1000

    print(f"   Time: {elapsed:.1f}ms")
    print(f"   Input size: {test_image.shape}")
    print(f"   Output size: {enhanced_clahe.shape}")

    # Test Method 2: Real-ESRGAN (if model available)
    model_path = "models/enhancement/RealESRGAN_x4plus.pth"
    if Path(model_path).exists():
        print("\n2. Testing Real-ESRGAN + CLAHE + Bilateral...")
        enhancer_esrgan = ImageEnhancer(
            method="realesrgan",
            realesrgan_model_path=model_path,
            device="cuda:0" if torch.cuda.is_available() else "cpu"
        )

        start = time.time()
        enhanced_esrgan = enhancer_esrgan.enhance(test_image)
        elapsed = (time.time() - start) * 1000

        print(f"   Time: {elapsed:.1f}ms")
        print(f"   Input size: {test_image.shape}")
        print(f"   Output size: {enhanced_esrgan.shape}")
        print(f"   Upscale: {test_image.shape[0]}x{test_image.shape[1]} → {enhanced_esrgan.shape[0]}x{enhanced_esrgan.shape[1]}")
    else:
        print(f"\n2. Real-ESRGAN model not found at: {model_path}")
        print("   Download with: wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth")

    print("\n" + "="*60)
    print("✓ ImageEnhancer tests complete")

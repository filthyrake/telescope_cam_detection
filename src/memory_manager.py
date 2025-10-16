"""
GPU Memory Manager - Graceful degradation for OOM scenarios (Issue #125)
Monitors GPU memory usage and provides progressive degradation strategies.
"""

import torch
import logging
import time
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class MemoryPressure(Enum):
    """GPU memory pressure levels"""
    NORMAL = "normal"      # < 75% usage
    HIGH = "high"          # 75-85% usage
    CRITICAL = "critical"  # 85-95% usage
    EXTREME = "extreme"    # > 95% usage


class MemoryManager:
    """
    Manage GPU memory pressure and provide graceful degradation strategies.

    Features:
    - Real-time GPU memory monitoring
    - Progressive memory pressure detection (normal/high/critical/extreme)
    - Automatic memory reduction strategies based on pressure level
    - Recovery tracking and hysteresis to prevent thrashing

    Related Issues: #98, #99, #110, #122, #125
    """

    def __init__(
        self,
        device: str = "cuda:0",
        warning_threshold: float = 0.75,  # 75% - start warning
        high_threshold: float = 0.85,     # 85% - high pressure
        critical_threshold: float = 0.95, # 95% - critical pressure
        check_interval: float = 1.0,      # Check every 1 second
        hysteresis: float = 0.05          # 5% hysteresis to prevent thrashing
    ):
        """
        Initialize Memory Manager.

        Args:
            device: CUDA device to monitor
            warning_threshold: Memory usage % to trigger warnings (0.0-1.0)
            high_threshold: Memory usage % for high pressure (0.0-1.0)
            critical_threshold: Memory usage % for critical pressure (0.0-1.0)
            check_interval: How often to check memory (seconds)
            hysteresis: Percentage hysteresis to prevent thrashing (0.0-1.0)
        """
        self.device = device
        self.warning_threshold = warning_threshold
        self.high_threshold = high_threshold
        self.critical_threshold = critical_threshold
        self.check_interval = check_interval
        self.hysteresis = hysteresis

        # State tracking
        self.current_pressure = MemoryPressure.NORMAL
        self.last_check_time = 0.0
        self.last_warning_time = 0.0
        self.warning_cooldown = 30.0  # Only warn every 30 seconds

        # Recovery tracking
        self.oom_events = 0
        self.recoveries = 0
        self.degradation_level = 0  # 0 = no degradation, higher = more aggressive

        # Check if CUDA is available
        self.cuda_available = torch.cuda.is_available()
        if not self.cuda_available:
            logger.warning("CUDA not available - Memory Manager running in CPU mode")

    def check_memory_pressure(self, force: bool = False) -> MemoryPressure:
        """
        Check current GPU memory pressure level.

        Args:
            force: Force check even if check_interval hasn't elapsed

        Returns:
            Current memory pressure level
        """
        # Skip check if CUDA not available
        if not self.cuda_available:
            return MemoryPressure.NORMAL

        # Rate limit checks unless forced
        current_time = time.time()
        if not force and (current_time - self.last_check_time) < self.check_interval:
            return self.current_pressure

        self.last_check_time = current_time

        try:
            # Get memory statistics
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            total = torch.cuda.get_device_properties(self.device).total_memory

            # Calculate usage as percentage using reserved memory.
            # We use reserved (not allocated) because:
            # 1. Reserved memory represents what PyTorch has claimed from the GPU
            # 2. PyTorch won't automatically release reserved memory without explicit cache clearing
            # 3. The difference (reserved - allocated) is freeable cache, which our degradation
            #    strategy addresses through cache clearing at high pressure levels
            # 4. Reserved memory is a better indicator of imminent OOM conditions
            usage_percent = reserved / total if total > 0 else 0.0

            # Determine pressure level with hysteresis
            # Use hysteresis to prevent rapid state changes (thrashing)
            if usage_percent >= self.critical_threshold:
                new_pressure = MemoryPressure.EXTREME
            elif usage_percent >= self.high_threshold:
                new_pressure = MemoryPressure.CRITICAL
            elif usage_percent >= self.warning_threshold:
                new_pressure = MemoryPressure.HIGH
            else:
                # Apply hysteresis when pressure is decreasing
                if self.current_pressure != MemoryPressure.NORMAL:
                    # Only return to normal if usage is below threshold - hysteresis
                    if usage_percent < (self.warning_threshold - self.hysteresis):
                        new_pressure = MemoryPressure.NORMAL
                    else:
                        new_pressure = self.current_pressure  # Stay in current state
                else:
                    new_pressure = MemoryPressure.NORMAL

            # Log pressure changes
            if new_pressure != self.current_pressure:
                logger.warning(
                    f"GPU memory pressure changed: {self.current_pressure.value} → {new_pressure.value} "
                    f"(usage: {usage_percent*100:.1f}%, allocated: {allocated/1e9:.2f}GB, "
                    f"reserved: {reserved/1e9:.2f}GB, total: {total/1e9:.2f}GB)"
                )
                self.current_pressure = new_pressure

            # Periodic warnings for sustained high pressure
            if new_pressure != MemoryPressure.NORMAL:
                if (current_time - self.last_warning_time) >= self.warning_cooldown:
                    logger.warning(
                        f"Sustained GPU memory pressure: {new_pressure.value} "
                        f"({usage_percent*100:.1f}% used, degradation level: {self.degradation_level})"
                    )
                    self.last_warning_time = current_time

            return new_pressure

        except Exception as e:
            logger.error(f"Failed to check GPU memory pressure: {e}")
            return MemoryPressure.NORMAL

    def reduce_memory_usage(self, level: MemoryPressure) -> Dict[str, Any]:
        """
        Reduce memory usage based on pressure level.

        Args:
            level: Current memory pressure level

        Returns:
            Dict of recommended settings to reduce memory usage
        """
        recommendations = {
            'clear_cache': False,
            'reduce_batch_size': False,
            'disable_frame_buffering': False,
            'reduce_input_size': False,
            'suggested_input_size': None,
            'suggested_batch_size': None,
        }

        if level == MemoryPressure.HIGH:
            # Level 1: Clear cache only
            recommendations['clear_cache'] = True
            logger.info("Memory reduction: Clearing GPU cache")

        elif level == MemoryPressure.CRITICAL:
            # Level 2: Clear cache + reduce batch size
            recommendations['clear_cache'] = True
            recommendations['reduce_batch_size'] = True
            recommendations['suggested_batch_size'] = 1
            logger.warning("Memory reduction: Clearing cache + reducing batch size to 1")

        elif level == MemoryPressure.EXTREME:
            # Level 3: Full degradation - reduce input size and batch
            recommendations['clear_cache'] = True
            recommendations['reduce_batch_size'] = True
            recommendations['disable_frame_buffering'] = True
            recommendations['reduce_input_size'] = True
            recommendations['suggested_input_size'] = (640, 640)
            recommendations['suggested_batch_size'] = 1
            logger.error(
                "Memory reduction: EXTREME pressure - reducing input size to 640x640, "
                "batch size = 1"
            )

            # Increase degradation level
            self.degradation_level += 1

        return recommendations

    def handle_oom_error(self) -> Dict[str, Any]:
        """
        Handle OOM error by clearing memory and providing recovery recommendations.

        Returns:
            Dict of recovery recommendations
        """
        self.oom_events += 1
        logger.error(f"GPU OOM detected (event #{self.oom_events}) - attempting recovery...")

        # Clear all GPU caches
        if self.cuda_available:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("GPU cache cleared")
            except Exception as e:
                logger.error(f"Failed to clear GPU cache: {e}")

        # Progressive degradation based on OOM frequency
        recommendations = {
            'clear_cache': True,
            'reduce_batch_size': True,
            'reduce_input_size': True,
            'cpu_fallback': False,
            'suggested_input_size': (640, 640),
            'suggested_batch_size': 1,
            'wait_time': 1.0,  # Wait 1 second before retrying
        }

        # If OOM events are frequent, recommend CPU fallback
        if self.oom_events >= 3:
            recommendations['cpu_fallback'] = True
            logger.error(
                f"Multiple OOM events detected ({self.oom_events}) - "
                "recommending CPU fallback for critical components"
            )

        # Increase degradation level
        self.degradation_level = min(self.degradation_level + 1, 5)  # Cap at 5

        return recommendations

    def record_recovery(self):
        """Record successful recovery from OOM/high memory pressure"""
        self.recoveries += 1
        logger.info(f"GPU memory recovery #{self.recoveries} successful")

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get current GPU memory statistics.

        Returns:
            Dict of memory statistics
        """
        stats = {
            'cuda_available': self.cuda_available,
            'current_pressure': self.current_pressure.value,
            'oom_events': self.oom_events,
            'recoveries': self.recoveries,
            'degradation_level': self.degradation_level,
        }

        if self.cuda_available:
            try:
                allocated = torch.cuda.memory_allocated(self.device)
                reserved = torch.cuda.memory_reserved(self.device)
                total = torch.cuda.get_device_properties(self.device).total_memory

                stats.update({
                    'allocated_mb': allocated / 1024 / 1024,
                    'allocated_gb': allocated / 1e9,
                    'reserved_mb': reserved / 1024 / 1024,
                    'reserved_gb': reserved / 1e9,
                    'total_mb': total / 1024 / 1024,
                    'total_gb': total / 1e9,
                    'usage_percent': (reserved / total * 100) if total > 0 else 0.0,
                    'allocated_percent': (allocated / total * 100) if total > 0 else 0.0,
                })
            except Exception as e:
                logger.error(f"Failed to get GPU memory stats: {e}")

        return stats

    def clear_cache(self):
        """Clear GPU cache to free memory"""
        if self.cuda_available:
            try:
                torch.cuda.empty_cache()
                logger.debug("GPU cache cleared")
            except Exception as e:
                logger.error(f"Failed to clear GPU cache: {e}")

    def reset_degradation(self):
        """Reset degradation level (call after successful operation at normal pressure)"""
        if self.degradation_level > 0:
            logger.info(f"Resetting degradation level from {self.degradation_level} to 0")
            self.degradation_level = 0

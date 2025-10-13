#!/usr/bin/env python3
"""
Camera Lag Monitoring Script

Monitors the latency difference between cameras to detect lag accumulation.
Run this for several minutes to observe if cam2 falls behind cam1.

Usage:
    python scripts/monitor_camera_lag.py [--interval 5] [--duration 600]
"""

import argparse
import time
import requests
import sys
from datetime import datetime
from typing import Dict, Any


def get_camera_stats() -> Dict[str, Any]:
    """Fetch camera statistics from the web server."""
    try:
        response = requests.get('http://localhost:8000/stats', timeout=2)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching stats: {e}", file=sys.stderr)
        return None


def calculate_lag(stats: Dict[str, Any]) -> Dict[str, float]:
    """Calculate lag metrics from stats."""
    if not stats or 'last_detection_times' not in stats:
        return None

    now = time.time()
    last_times = stats['last_detection_times']

    if 'cam1' not in last_times or 'cam2' not in last_times:
        return None

    cam1_age = now - last_times['cam1']
    cam2_age = now - last_times['cam2']
    lag_diff = abs(cam1_age - cam2_age)

    return {
        'timestamp': now,
        'cam1_age': cam1_age,
        'cam2_age': cam2_age,
        'lag_diff': lag_diff,
        'cam2_behind': cam2_age > cam1_age
    }


def format_lag_report(lag_data: Dict[str, float]) -> str:
    """Format lag data for display."""
    ts = datetime.fromtimestamp(lag_data['timestamp']).strftime('%H:%M:%S')
    cam1 = lag_data['cam1_age']
    cam2 = lag_data['cam2_age']
    diff = lag_data['lag_diff']
    behind = "cam2" if lag_data['cam2_behind'] else "cam1"

    status = "⚠️  LAGGING" if diff > 5.0 else "✓ OK" if diff < 1.0 else "~ MILD"

    return f"[{ts}] {status} | cam1: {cam1:5.1f}s | cam2: {cam2:5.1f}s | diff: {diff:5.1f}s | {behind} behind"


def monitor_lag(interval: int = 5, duration: int = 600):
    """
    Monitor camera lag continuously.

    Args:
        interval: Seconds between checks
        duration: Total monitoring duration in seconds (0 = infinite)
    """
    print(f"🐕 Camera Lag Monitor")
    print(f"   Interval: {interval}s")
    print(f"   Duration: {duration}s" if duration > 0 else "   Duration: continuous (Ctrl+C to stop)")
    print(f"   Timestamp format: HH:MM:SS")
    print()
    print("Status Guide:")
    print("  ✓ OK      - Cameras in sync (< 1s difference)")
    print("  ~ MILD    - Minor lag (1-5s difference)")
    print("  ⚠️  LAGGING - Significant lag (> 5s difference)")
    print()
    print("=" * 80)

    start_time = time.time()
    max_lag_seen = 0.0
    lag_samples = []

    try:
        while True:
            elapsed = time.time() - start_time

            # Check duration limit
            if duration > 0 and elapsed >= duration:
                break

            # Get stats and calculate lag
            stats = get_camera_stats()
            if stats:
                lag_data = calculate_lag(stats)
                if lag_data:
                    print(format_lag_report(lag_data))

                    # Track max lag
                    if lag_data['lag_diff'] > max_lag_seen:
                        max_lag_seen = lag_data['lag_diff']

                    lag_samples.append(lag_data['lag_diff'])
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠ No detection data available")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Failed to fetch stats")

            time.sleep(interval)

    except KeyboardInterrupt:
        print()
        print("Monitoring stopped by user")

    # Print summary
    print("=" * 80)
    print()
    print("📊 Summary:")
    print(f"   Duration: {elapsed:.0f}s")
    print(f"   Samples: {len(lag_samples)}")
    print(f"   Max lag observed: {max_lag_seen:.1f}s")

    if lag_samples:
        avg_lag = sum(lag_samples) / len(lag_samples)
        print(f"   Average lag: {avg_lag:.1f}s")

        # Calculate trend (is lag increasing?)
        if len(lag_samples) >= 4:
            first_half_avg = sum(lag_samples[:len(lag_samples)//2]) / (len(lag_samples)//2)
            second_half_avg = sum(lag_samples[len(lag_samples)//2:]) / (len(lag_samples) - len(lag_samples)//2)
            trend = second_half_avg - first_half_avg

            if trend > 1.0:
                print(f"   Trend: ⚠️  INCREASING (+{trend:.1f}s) - Lag is accumulating!")
            elif trend < -1.0:
                print(f"   Trend: ✓ DECREASING ({trend:.1f}s) - Lag is improving")
            else:
                print(f"   Trend: ~ STABLE ({trend:+.1f}s)")

    print()

    if max_lag_seen > 10.0:
        print("⚠️  WARNING: Significant lag detected (>10s)")
        print("   Possible causes:")
        print("   - Neolink buffer_size too large")
        print("   - Network interruptions")
        print("   - RTSP buffering in OpenCV")
        print("   - Processing bottleneck")


def main():
    parser = argparse.ArgumentParser(
        description='Monitor camera lag and detect accumulation'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=5,
        help='Seconds between checks (default: 5)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=600,
        help='Total monitoring duration in seconds, 0=infinite (default: 600)'
    )

    args = parser.parse_args()
    monitor_lag(interval=args.interval, duration=args.duration)


if __name__ == '__main__':
    main()

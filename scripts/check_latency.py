#!/usr/bin/env python3
"""Quick script to check current system latency from WebSocket."""

import asyncio
import websockets
import json
import sys

async def check_latency():
    uri = "ws://localhost:8000/ws/detections"
    latencies = []

    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket. Collecting latency samples...")

            for i in range(20):  # Collect 20 samples
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(message)

                if 'total_latency_ms' in data:
                    latency = data['total_latency_ms']
                    latencies.append(latency)
                    print(f"Sample {i+1}: {latency:.1f}ms")

            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                min_latency = min(latencies)
                max_latency = max(latencies)

                print(f"\n=== Latency Summary ===")
                print(f"Samples: {len(latencies)}")
                print(f"Average: {avg_latency:.1f}ms")
                print(f"Min: {min_latency:.1f}ms")
                print(f"Max: {max_latency:.1f}ms")

                if avg_latency > 1000:
                    print(f"\n⚠️  WARNING: Average latency is {avg_latency:.0f}ms (>1000ms)")
                elif avg_latency > 500:
                    print(f"\n⚠️  High latency: {avg_latency:.0f}ms (target <250ms)")
                else:
                    print(f"\n✓ Latency is acceptable")
            else:
                print("No latency data received")

    except asyncio.TimeoutError:
        print("Timeout waiting for detection messages")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(check_latency())

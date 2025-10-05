#!/usr/bin/env python3
"""Quick status check of running system."""

import requests
import time

# Check health endpoint
try:
    response = requests.get("http://localhost:8000/health", timeout=2)
    print("Health check:", response.json())
except Exception as e:
    print(f"Health check failed: {e}")

# Check stats
try:
    response = requests.get("http://localhost:8000/stats", timeout=2)
    print("Stats:", response.json())
except Exception as e:
    print(f"Stats check failed: {e}")

print("\nSystem appears to be running. Check GPU activity:")
print("Run: nvidia-smi")

#!/usr/bin/env python3
"""
Evaluate trained custom model on test images or live camera feed.
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import cv2
import torch
from ultralytics import YOLO
from stream_capture import create_rtsp_url, RTSPStreamCapture
from queue import Queue


def evaluate_on_images(model_path: str, images_dir: str, conf_threshold: float = 0.25):
    """
    Evaluate model on a directory of images.

    Args:
        model_path: Path to trained model
        images_dir: Directory containing test images
        conf_threshold: Confidence threshold
    """
    print(f"📊 Evaluating model: {model_path}")
    print(f"📂 Test images: {images_dir}\n")

    model = YOLO(model_path)

    images_path = Path(images_dir)
    image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))

    if not image_files:
        print(f"❌ No images found in {images_dir}")
        return

    print(f"Found {len(image_files)} images\n")

    for img_file in image_files:
        print(f"Processing: {img_file.name}")

        # Run inference
        results = model(str(img_file), conf=conf_threshold)[0]

        # Display results
        if results.boxes is not None and len(results.boxes) > 0:
            print(f"  ✅ Found {len(results.boxes)} detections:")
            for box in results.boxes:
                class_id = int(box.cls)
                conf = float(box.conf)
                class_name = results.names[class_id]
                print(f"     - {class_name}: {conf:.2f}")

            # Show annotated image
            annotated = results.plot()
            cv2.imshow('Detection Results', annotated)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
        else:
            print(f"  ⚠️  No detections")

    cv2.destroyAllWindows()
    print(f"\n✅ Evaluation complete")


def evaluate_live(
    model_path: str,
    camera_ip: str = "10.0.8.18",
    username: str = "admin",
    password: str = "5326jbbD",
    conf_threshold: float = 0.25
):
    """
    Evaluate model on live camera feed.

    Args:
        model_path: Path to trained model
        camera_ip: Camera IP address
        username: Camera username
        password: Camera password
        conf_threshold: Confidence threshold
    """
    print(f"📊 Evaluating model: {model_path}")
    print(f"📹 Live feed from: {camera_ip}\n")
    print("Press 'q' to quit\n")

    model = YOLO(model_path)

    # Connect to camera
    rtsp_url = create_rtsp_url(camera_ip, username, password, "main")
    frame_queue = Queue(maxsize=2)

    capture = RTSPStreamCapture(rtsp_url, frame_queue)

    if not capture.start():
        print("❌ Failed to connect to camera")
        return

    print("✅ Connected to camera\n")

    try:
        while True:
            if not frame_queue.empty():
                frame_data = frame_queue.get()
                frame = frame_data['frame']

                # Run inference
                results = model(frame, conf=conf_threshold, verbose=False)[0]

                # Draw detections
                annotated = results.plot()

                # Add stats
                detection_count = len(results.boxes) if results.boxes is not None else 0
                cv2.putText(
                    annotated,
                    f"Detections: {detection_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                cv2.imshow('Custom Model - Live Evaluation', annotated)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\n⚠️  Stopped by user")

    finally:
        capture.stop()
        cv2.destroyAllWindows()
        print("\n✅ Evaluation complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate custom telescope detection model")
    parser.add_argument("--model", default="models/telescope_custom.pt",
                       help="Path to trained model")
    parser.add_argument("--mode", choices=["images", "live"], default="live",
                       help="Evaluation mode")
    parser.add_argument("--images", default="training/datasets/telescope_equipment/images/val",
                       help="Test images directory (for images mode)")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Confidence threshold")
    parser.add_argument("--camera-ip", default="10.0.8.18",
                       help="Camera IP (for live mode)")
    parser.add_argument("--username", default="admin",
                       help="Camera username (for live mode)")
    parser.add_argument("--password", default="5326jbbD",
                       help="Camera password (for live mode)")

    args = parser.parse_args()

    if args.mode == "images":
        evaluate_on_images(
            model_path=args.model,
            images_dir=args.images,
            conf_threshold=args.conf
        )
    else:
        evaluate_live(
            model_path=args.model,
            camera_ip=args.camera_ip,
            username=args.username,
            password=args.password,
            conf_threshold=args.conf
        )

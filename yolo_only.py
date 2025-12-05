"""
Quick YOLOv8 inference demo for confidence/visual check (no robot, no hand-eye).

Usage examples
--------------
1) Webcam test with your trained weights:
   python yolo_only.py --weights yolov8n.pt --source 0

2) Specific image or video file:
   python yolo_only.py --weights best.pt --source path/to/file.jpg

Press q to quit the display window.
"""

import argparse
import time

import cv2
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 quick visualizer")
    p.add_argument("--weights", type=str, default="best.pt", help="Path to YOLO weights")
    p.add_argument("--source", type=str, default="0", help="Camera index or image/video path")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference size")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(r"D:\yolo\runs\detect\train\weights\best.pt")

    # Decide capture source
    cap = None
    is_cam = args.source.isdigit()
    if is_cam:
        cap = cv2.VideoCapture(int(args.source))
        if not cap.isOpened():
            raise SystemExit(f"Cannot open camera {args.source}")
    else:
        # Let YOLO handle file sources directly if not webcam
        pass

    fps_avg = 0.0
    while True:
        if is_cam:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame")
                break
            start = time.time()
            results = model.predict(source=frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
            vis = results[0].plot()
            fps = 1.0 / (time.time() - start + 1e-6)
            fps_avg = 0.8 * fps_avg + 0.2 * fps if fps_avg else fps
            cv2.putText(vis, f"FPS: {fps_avg:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("YOLO only", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # For image/video path we can delegate to YOLO's built-in show
            model.predict(source=args.source, conf=args.conf, imgsz=args.imgsz, show=True)
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""
Quick YOLOv8 inference + ZXing decode (no robot, no hand-eye).

Usage examples
--------------
1) Webcam test with your trained weights:
   python yolo_only.py --weights yolov8n.pt --source 0 --decode

2) Image or video file:
   python yolo_only.py --weights best.pt --source path/to/file.mp4 --decode

Press q to quit the display window.
"""

import argparse
import os
import tempfile
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

try:
    import zxing  # type: ignore

    ZXING_AVAILABLE = True
except Exception:
    ZXING_AVAILABLE = False


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 quick visualizer with optional ZXing decode")
    p.add_argument("--weights", type=str, default="best.pt", help="Path to YOLO weights")
    p.add_argument("--source", type=str, default="0", help="Camera index or image/video path")
    p.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference size")
    p.add_argument("--decode", action="store_true", help="Decode barcode/QR using ZXing and print text")
    return p.parse_args()


def decode_with_zxing(reader, frame):
    """Decode one frame via ZXing by writing a temp PNG. Returns parsed text or None."""
    tmp_path = os.path.join(tempfile.gettempdir(), f"zxing_{int(time.time()*1000)}.png")
    cv2.imwrite(tmp_path, frame)
    result = reader.decode(tmp_path)
    os.remove(tmp_path)
    if result is None:
        return None
    # Some bindings expose parsed/parsed_text/raw_text attributes; fallback to text
    for attr in ("parsed", "parsed_text", "raw", "raw_text", "text"):
        if hasattr(result, attr):
            val = getattr(result, attr)
            if val:
                return val
    return None


def main():
    args = parse_args()
    model = YOLO(r"D:\yolo\runs\detect\train\weights\best.pt")

    reader = None
    if args.decode:
        if ZXING_AVAILABLE:
            reader = zxing.BarCodeReader()
            print("[ZXING] decoder enabled")
        else:
            print("[ZXING] zxing not importable; decode disabled")

    # Decide capture source
    is_cam = args.source.isdigit()
    cap = None
    frame_source = None

    if is_cam:
        cap = cv2.VideoCapture(int(args.source))
        if not cap.isOpened():
            raise SystemExit(f"Cannot open camera {args.source}")
        frame_source = "camera"
    else:
        path = Path(args.source)
        if not path.exists():
            raise SystemExit(f"Source not found: {args.source}")
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            frame_source = "video"
        else:
            img = cv2.imread(str(path))
            if img is None:
                raise SystemExit(f"Cannot read image: {args.source}")
            frame_source = "image"
            frames = [img]

    fps_avg = 0.0
    frame_iter = None
    if frame_source in ("camera", "video"):
        frame_iter = iter(int, 1)  # endless; break on read failure
    else:
        frame_iter = frames

    for _ in frame_iter:
        if frame_source in ("camera", "video"):
            ok, frame = cap.read()
            if not ok:
                print("End of stream" if frame_source == "video" else "Failed to read frame")
                break
        else:  # single image
            frame = _

        start = time.time()
        results = model.predict(source=frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
        vis = results[0].plot()

        if reader is not None:
            decoded_text = decode_with_zxing(reader, frame)
            if decoded_text:
                print(f"[ZXING] {decoded_text}")
                cv2.putText(vis, decoded_text[:60], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        fps = 1.0 / (time.time() - start + 1e-6)
        fps_avg = 0.8 * fps_avg + 0.2 * fps if fps_avg else fps
        cv2.putText(vis, f"FPS: {fps_avg:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("YOLO only", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

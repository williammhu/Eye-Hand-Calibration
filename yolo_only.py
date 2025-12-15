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
import zxing

# Keep a single reader to avoid repeated init cost.
ZXING_READER = zxing.BarCodeReader()


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 quick visualizer with optional ZXing decode")
    p.add_argument("--weights", type=str, default=r"D:\yolo\runs\detect\train\weights\best.pt", help="Path to YOLO weights")
    p.add_argument("--source", type=str, default="0", help="Camera index or image/video path")
    p.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference size")
    p.add_argument("--cam-width", type=int, default=1280, help="Requested camera width (pixels)")
    p.add_argument("--cam-height", type=int, default=720, help="Requested camera height (pixels)")
    p.add_argument("--cam-fps", type=int, default=30, help="Requested camera FPS")
    p.add_argument("--decode", action="store_true", help="Decode barcode/QR using ZXing and print text")
    return p.parse_args()


def decode_with_zxing(frame):
    """
    Decode one frame/crop via ZXing.
    ZXing Python binding currently expects a file path, so we temp-save the crop as PNG.
    Returns parsed text or None.
    """
    ok, buf = cv2.imencode(".png", frame)
    if not ok:
        return None

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(buf.tobytes())
            tmp_path = tmp.name
        result = ZXING_READER.decode(tmp_path, try_harder=True)
        if result is None:
            return None
        return getattr(result, "parsed", None) or getattr(result, "raw", None)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    args = parse_args()
    model = YOLO(args.weights)

    if args.decode:
        print("[ZXING] decoder enabled", flush=True)

    # Decide capture source
    is_cam = args.source.isdigit()
    cap = None
    frame_source = None

    if is_cam:
        cam_index = int(args.source)
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            raise SystemExit(f"Cannot open camera {args.source}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
        cap.set(cv2.CAP_PROP_FPS, args.cam_fps)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[CAM] requested {args.cam_width}x{args.cam_height}@{args.cam_fps} -> got {actual_w}x{actual_h}@{actual_fps:.1f}", flush=True)
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

        if args.decode:
            h, w = frame.shape[:2]
            boxes = results[0].boxes
            if boxes is not None and boxes.xyxy is not None:
                for box in boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = box.astype(int)
                    x1 = max(0, min(x1, w - 1))
                    x2 = max(0, min(x2, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    y2 = max(0, min(y2, h - 1))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    roi = frame[y1:y2, x1:x2]
                    roi_big = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
                    gray = cv2.cvtColor(roi_big, cv2.COLOR_BGR2GRAY)
                    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    decoded_text = decode_with_zxing(bw)
                    if decoded_text:
                        print(f"[ZXING] {decoded_text}")
                        cv2.putText(
                            vis,
                            decoded_text[:60],
                            (x1 + 5, max(20, y1 + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 200, 255),
                            2,
                        )

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

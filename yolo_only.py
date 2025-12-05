"""
Quick YOLOv8 inference + pyzbar decode (no robot, no hand-eye).

Usage examples
--------------
1) Webcam test with your trained weights:
   python yolo_only.py --weights yolov8n.pt --source 0 --decode

2) Image or video file:
   python yolo_only.py --weights best.pt --source path/to/file.mp4 --decode
python yolo_only.py --decode
Press q to quit the display window.
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO
from pyzbar import pyzbar


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 quick visualizer with optional pyzbar decode")
    p.add_argument("--weights", type=str, default=r"D:\yolo\runs\detect\train\weights\best.pt", help="Path to YOLO weights")
    p.add_argument("--source", type=str, default="0", help="Camera index or image/video path")
    p.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference size")
    p.add_argument("--decode", action="store_true", help="Decode barcode/QR using pyzbar and print text")
    p.add_argument(
        "--allow-pdf417",
        action="store_true",
        help="Include PDF417 symbology in decoding (disabled by default to avoid zbar assertion warnings)",
    )
    return p.parse_args()


def decode_with_pyzbar(frame, allow_pdf417=False):
    """
    Decode one frame via pyzbar. Returns parsed text or None.
    PDF417 decoding is disabled by default to avoid zbar assertion warnings on some frames.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Limit symbologies to avoid zbar's PDF417 assertion; user can re-enable via flag.
    symbols = None
    if not allow_pdf417:
        symbols = [
            pyzbar.ZBarSymbol.CODE128,
            pyzbar.ZBarSymbol.CODE39,
            pyzbar.ZBarSymbol.EAN13,
            pyzbar.ZBarSymbol.EAN8,
            pyzbar.ZBarSymbol.UPCA,
            pyzbar.ZBarSymbol.UPCE,
            pyzbar.ZBarSymbol.I25,
            pyzbar.ZBarSymbol.DATABAR,
            pyzbar.ZBarSymbol.DATABAR_EXP,
            pyzbar.ZBarSymbol.CODABAR,
            pyzbar.ZBarSymbol.QRCODE,
        ]

    decoded = pyzbar.decode(gray, symbols=symbols)
    for obj in decoded:
        try:
            text = obj.data.decode("utf-8", errors="ignore")
        except Exception:
            continue
        if text:
            return text
    return None


def main():
    args = parse_args()
    model = YOLO(args.weights)

    if args.decode:
        print("[PYZBAR] decoder enabled")

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
                    crop = frame[y1:y2, x1:x2]
                    decoded_text = decode_with_pyzbar(crop, allow_pdf417=args.allow_pdf417)
                    if decoded_text:
                        print(f"[PYZBAR] {decoded_text}")
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

"""
Minimal camera sanity-check using the same open logic as Hand_Eye_Calibration.py.
Shows the live feed; press 'q' to exit.
"""

import argparse
import cv2


def parse_args():
    p = argparse.ArgumentParser(description="Quick camera test (index or URL)")
    p.add_argument("--source", type=str, default="0", help="Camera index like '0'/'1' or URL/path for virtual/phone cams")
    p.add_argument("--cam-width", type=int, default=1280, help="Requested camera width")
    p.add_argument("--cam-height", type=int, default=720, help="Requested camera height")
    p.add_argument("--cam-fps", type=int, default=30, help="Requested camera FPS")
    return p.parse_args()


def open_camera(src_str: str, w: int, h: int, fps: int) -> cv2.VideoCapture:
    # Accept numeric index or string path/URL
    src = int(src_str) if src_str.isdigit() else src_str
    cap = cv2.VideoCapture(src)
    if src_str.isdigit():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        raise SystemExit(f"Could not open camera source {src_str}")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[CAM] {src_str} -> {actual_w}x{actual_h}@{actual_fps:.1f}")
    return cap


def main():
    args = parse_args()
    cap = open_camera(args.source, args.cam_width, args.cam_height, args.cam_fps)
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame")
            break
        cv2.imshow("cam_test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

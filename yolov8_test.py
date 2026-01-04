"""
YOLOv8 + hand-eye calibration (planar homography) pick-and-place demo.

Workflow
--------
1) Run Hand_Eye_Calibration.py --mode calibrate (or calibration_test.py) to generate
   save_parms/homography.npy / homography_inv.npy.
2) Place objects on the same plane used for calibration.
3) Start this script. It will:
   - detect objects with YOLOv8,
   - map the pixel centre to robot XY via the homography,
   - send the target to the Freenove arm client.

The detection loop stays in the main thread (so OpenCV windows work on Windows);
robot motion runs in a background thread and processes one target at a time.
Press q to quit.
python yolov8_test.py --parms-dir save_parms
"""

from __future__ import annotations

import argparse
import queue
import threading
from typing import Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from Hand_Eye_Calibration import HomographyResult
from freenove_arm import FreenoveArmClient


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
def pixel_to_robot(pixel: Sequence[float], homography: np.ndarray) -> Tuple[float, float]:
    """Map (u, v) pixel to robot XY using a 3x3 homography."""
    pt = np.array([pixel[0], pixel[1], 1.0], dtype=np.float64)
    mapped = homography @ pt
    mapped /= mapped[2]
    return float(mapped[0]), float(mapped[1])


def select_best_box(result, allowed: Optional[Set[str]]) -> Optional[int]:
    """
    Return index of the highest-confidence detection that matches the
    allowed class names (or any class if ``allowed`` is ``None``).
    """
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    best_idx = None
    best_conf = 0.0
    for i in range(len(boxes)):
        conf = float(boxes.conf[i])
        cls_id = int(boxes.cls[i])
        cls_name = result.names[cls_id]
        if allowed and cls_name not in allowed:
            continue
        if conf > best_conf:
            best_conf = conf
            best_idx = i
    return best_idx


# --------------------------------------------------------------------------- #
# Robot worker
# --------------------------------------------------------------------------- #
def robot_worker(args: argparse.Namespace, target_queue: "queue.Queue", busy_flag: threading.Event) -> None:
    """
    Blocking loop that consumes targets from the queue and drives the robot.
    Each target is a tuple (x_mm, y_mm, z_mm, cls_id).
    """
    try:
        with FreenoveArmClient(
            host=args.host,
            port=args.port,
            dry_run=args.dry_run,
            # Always enable motors for the full run to keep power on as requested.
            auto_enable=True,
        ) as arm:
            if args.skip_enable:
                print("[robot] --skip-enable ignored; motors are kept enabled.")
            # Mirror quick_move/Hand_Eye_Calibration startup: home once and optionally set clearance.
            if args.home_first:
                arm.return_to_sensor_point(1)
                arm.wait(0.5)
            if args.ground_clearance is not None:
                arm.set_ground_clearance(args.ground_clearance)
                arm.wait(0.1)

            calib_point = (args.calib_x, args.calib_y, args.calib_z)
            print("[robot] connected to arm")
            while True:
                item = target_queue.get()
                if item is None:
                    break

                x, y, z, cls_id = item
                print(f"[robot] target ({x:.1f}, {y:.1f}, {z:.1f}) cls={cls_id}")

                # Move to target, hold, then return to calibration point. Stay powered the whole time.
                arm.move_to(x, y, z, speed=args.speed)
                arm.wait(args.hold_time)
                arm.move_to(*calib_point, speed=args.speed)
                arm.wait(args.settle)

                busy_flag.clear()
    except Exception as exc:  # pragma: no cover - hardware path
        print(f"[robot] error: {exc}")
        busy_flag.clear()


# --------------------------------------------------------------------------- #
# Detection loop (runs in main thread)
# --------------------------------------------------------------------------- #
def detection_loop(args: argparse.Namespace) -> None:
    try:
        homography = HomographyResult.load(args.parms_dir).homography
    except FileNotFoundError:
        raise SystemExit(
            f"Homography files not found in '{args.parms_dir}'. "
            "Please run Hand_Eye_Calibration.py --mode calibrate first."
        )

    model = YOLO(args.weights)
    # Use the same DirectShow capture path as Hand_Eye_Calibration for Windows stability.
    cap = cv2.VideoCapture(args.camera_id, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise SystemExit("Could not open camera")

    allowed = set(args.classes) if args.classes else None
    target_queue: queue.Queue = queue.Queue(maxsize=1)
    busy_flag = threading.Event()
    detection_triggered = False  # stop detection after first successful target

    robot_thread = threading.Thread(
        target=robot_worker,
        args=(args, target_queue, busy_flag),
        daemon=True,
    )
    robot_thread.start()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[vision] failed to read frame")
                break

            if busy_flag.is_set():
                cv2.putText(frame, "Robot moving...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("yolo + hand-eye", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                continue

            # Once a target has been sent and the robot finished, stop detection.
            if detection_triggered and not busy_flag.is_set():
                cv2.putText(
                    frame,
                    "Cycle complete - detection stopped",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("yolo + hand-eye", frame)
                cv2.waitKey(1)
                break

            result = model.predict(
                source=frame,
                conf=args.conf,
                verbose=False,
            )[0]

            best_idx = select_best_box(result, allowed)
            status = "No target"

            if best_idx is not None:
                xyxy = result.boxes.xyxy[best_idx].cpu().numpy()
                cls_id = int(result.boxes.cls[best_idx])
                conf = float(result.boxes.conf[best_idx])
                cx = float((xyxy[0] + xyxy[2]) / 2.0)
                cy = float((xyxy[1] + xyxy[3]) / 2.0)
                rx, ry = pixel_to_robot((cx, cy), homography)

                # draw overlay
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"{result.names[cls_id]} {conf:.2f}",
                    (int(xyxy[0]), int(xyxy[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"robot XY=({rx:.1f},{ry:.1f})",
                    (int(xyxy[0]), int(xyxy[1]) + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                if (not busy_flag.is_set()) and conf >= args.min_pick_conf:
                    target_queue.put((rx, ry, args.z_height, cls_id))
                    busy_flag.set()
                    detection_triggered = True
                    status = "Target locked. Robot moving."

            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("yolo + hand-eye", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        target_queue.put(None)
        robot_thread.join(timeout=5)
        cap.release()
        cv2.destroyAllWindows()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine YOLOv8 detection with homography-based hand-eye calibration")
    parser.add_argument("--weights", type=str, default=r"D:\yolo\runs\detect\train\weights\best.pt", help="YOLO model weights")
    parser.add_argument("--camera-id", type=int, default=2, help="OpenCV camera index")
    parser.add_argument("--conf", type=float, default=0.5, help="YOLO detection confidence threshold")
    parser.add_argument("--min-pick-conf", type=float, default=0.5, help="Minimum confidence to send a pick command")
    parser.add_argument("--classes", nargs="*", help="Optional class name filter for picking")
    parser.add_argument("--parms-dir", type=str, default="save_parms", help="Directory containing homography.npy")
    parser.add_argument("--z-height", type=float, default=70.0, help="Table Z height for picking (mm)")
    parser.add_argument("--speed", type=int, default=50, help="Robot move speed hint")
    parser.add_argument("--hold-time", type=float, default=5.0, help="Pause at target position before returning (s)")
    parser.add_argument("--settle", type=float, default=0.5, help="Pause after return move (s)")
    parser.add_argument("--host", type=str, default="10.149.65.232", help="Robot IP address")
    parser.add_argument("--port", type=int, default=5000, help="Robot TCP port")
    parser.add_argument("--dry-run", action="store_true", help="Print commands instead of sending to robot")
    parser.add_argument("--skip-enable", action="store_true", help="(Ignored) Motors are always enabled for safety")
    parser.add_argument(
        "--home-first",
        dest="home_first",
        action="store_true",
        default=True,
        help="Send S10 F1 once after enabling motors (recommended)",
    )
    parser.add_argument(
        "--no-home-first",
        dest="home_first",
        action="store_false",
        help="Skip the homing move after enabling motors",
    )
    parser.add_argument(
        "--ground-clearance",
        type=float,
        default=None,
        help="Optional ground clearance height to set via S3 (mm)",
    )
    parser.add_argument("--calib-x", type=float, default=0.0, help="Calibration/home X to return after move (mm)")
    parser.add_argument("--calib-y", type=float, default=200.0, help="Calibration/home Y to return after move (mm)")
    parser.add_argument("--calib-z", type=float, default=90.0, help="Calibration/home Z to return after move (mm)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detection_loop(args)


if __name__ == "__main__":
    main()

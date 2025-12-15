"""
Planar hand–eye calibration script for the Freenove robot arm.

This version speaks the same TCP text protocol as the Freenove desktop app,
via :class:`freenove_arm.FreenoveArmClient`.  It estimates a 2D homography
between camera pixels and the robot table plane using an ArUco marker attached
to the gripper.
树莓派终端输入：
cd Freenove_Robot_Arm_Kit_for_Raspberry_Pi/Server/Code
sudo python main.py
cmd 输入： ping -4 raspberrypi.local
电脑测试：python calibration.py --mode calibrate --host 127.0.0.1 --port 5000 --dry-run
机械臂运行： python calibration.py --mode calibrate --host 10.149.65.232 --port 5000 --source 0
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from freenove_arm import FreenoveArmClient

# Supported ArUco/AprilTag dictionaries by name.
SUPPORTED_ARUCO_NAMES = [
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_4X4_250",
    "DICT_4X4_1000",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_5X5_250",
    "DICT_5X5_1000",
    "DICT_6X6_50",
    "DICT_6X6_100",
    "DICT_6X6_250",
    "DICT_6X6_1000",
    "DICT_7X7_50",
    "DICT_7X7_100",
    "DICT_7X7_250",
    "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
    "DICT_APRILTAG_16h5",
    "DICT_APRILTAG_25h9",
    "DICT_APRILTAG_36h10",
    "DICT_APRILTAG_36h11",
]
ARUCO_DICT_NAMES: Dict[str, int] = {name: getattr(cv2.aruco, name) for name in SUPPORTED_ARUCO_NAMES}

DEFAULT_ARUCO_NAME = "DICT_5X5_100"
MARKER_LENGTH_METERS = 0.02


@dataclass
class CameraConfig:
    """
    Camera capture settings.

    `source` matches OpenCV's VideoCapture argument: a digit string like "0" or
    an explicit path/URL for virtual/phone cameras (e.g. DroidCam, RTSP, MJPEG).
    """
    source: str = "0"
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass
class HomographyResult:
    homography: np.ndarray
    inverse: np.ndarray

    def save(self, directory: str = "save_parms") -> None:
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, "homography.npy"), self.homography)
        np.save(os.path.join(directory, "homography_inv.npy"), self.inverse)

    @classmethod
    def load(cls, directory: str = "save_parms") -> "HomographyResult":
        h_path = os.path.join(directory, "homography.npy")
        inv_path = os.path.join(directory, "homography_inv.npy")
        homography = np.load(h_path)
        inverse = np.load(inv_path)
        return cls(homography=homography, inverse=inverse)


class PlaneCalibrator:
    def __init__(self, camera_cfg: CameraConfig, dictionary_name: str = DEFAULT_ARUCO_NAME):
        if dictionary_name != "auto" and dictionary_name not in SUPPORTED_ARUCO_NAMES:
            raise ValueError(f"Unknown dictionary {dictionary_name}")

        self.cv2 = cv2
        self.aruco = cv2.aruco
        self.aruco_dicts = ARUCO_DICT_NAMES
        self.np = np

        self.camera_cfg = camera_cfg
        self.parameters = self.aruco.DetectorParameters()
        # Prepare one or many detectors depending on the CLI flag.
        target_names = SUPPORTED_ARUCO_NAMES if dictionary_name == "auto" else [dictionary_name]
        self.detectors: List[Tuple[str, object, object]] = []
        for name in target_names:
            d = self.aruco.getPredefinedDictionary(self.aruco_dicts[name])
            self.detectors.append((name, d, self.aruco.ArucoDetector(d, self.parameters)))
        self.active_dict: Optional[str] = None
        self.camera = self._open_camera(camera_cfg)

    def _open_camera(self, cfg: CameraConfig) -> object:
        # Accept both numeric indexes and string paths/URLs (phone cams, rtsp, etc.)
        src: int | str = int(cfg.source) if str(cfg.source).isdigit() else cfg.source
        cap = self.cv2.VideoCapture(src)
        if str(cfg.source).isdigit():
            cap.set(self.cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
            cap.set(self.cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
            cap.set(self.cv2.CAP_PROP_FPS, cfg.fps)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera source {cfg.source}")

        # Log the actual negotiated resolution/FPS to help diagnose virtual cams
        actual_w = int(cap.get(self.cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(self.cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(self.cv2.CAP_PROP_FPS)
        print(f"[CAM] {cfg.source} -> {actual_w}x{actual_h}@{actual_fps:.1f}")
        return cap

    def read_frame(self) -> np.ndarray:
        ok, frame = self.camera.read()
        if not ok:
            raise RuntimeError("Failed to read from camera")
        return frame

    def detect_marker(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)

        corners = ids = None
        rejected = None
        for name, _dict, detector in self.detectors:
            corners, ids, rejected = detector.detectMarkers(gray)
            if ids is not None and len(ids) > 0:
                self.active_dict = name
                break

        if ids is None or len(ids) == 0:
            return frame, None

        self.aruco.drawDetectedMarkers(frame, corners, ids)
        focal_length = max(self.camera_cfg.width, self.camera_cfg.height)
        camera_matrix = self.np.array(
            [[focal_length, 0, self.camera_cfg.width / 2], [0, focal_length, self.camera_cfg.height / 2], [0, 0, 1]],
            dtype=self.np.float32,
        )
        dist_coeffs = self.np.zeros((1, 5))
        rvecs, tvecs, _ = self.aruco.estimatePoseSingleMarkers(
            corners, MARKER_LENGTH_METERS, camera_matrix, dist_coeffs
        )
        for rvec, tvec in zip(rvecs, tvecs):
            self.cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH_METERS)

        corner = corners[0][0]
        center = corner.mean(axis=0)
        self.cv2.circle(frame, tuple(center.astype(int)), 5, (0, 0, 255), -1)
        label = f"{self.active_dict or 'Aruco'} center"
        self.cv2.putText(
            frame,
            label,
            (int(center[0]) + 5, int(center[1]) - 5),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        return frame, center

    def collect_correspondences(
        self,
        robot_points: Sequence[Tuple[float, float, float]],
        robot: FreenoveArmClient,
        settle_time: float = 1.0,
        step_mode: bool = False,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        np_module = self.np
        image_points: List[np.ndarray] = []
        robot_xy_points: List[np.ndarray] = []

        for idx, (x, y, z) in enumerate(robot_points):
            if step_mode:
                input(f"[{idx + 1}/{len(robot_points)}] Press Enter to move to ({x}, {y}, {z}) ...")
            print(f"Moving to calibration point {idx + 1}/{len(robot_points)}: ({x}, {y}, {z})")
            dwell_ms = int(settle_time * 1000)
            robot.move_to(x, y, z, dwell_ms=dwell_ms)
            robot.wait(settle_time)

            frame = self.read_frame()
            vis, center = self.detect_marker(frame)
            self.cv2.imshow("calibration", vis)
            self.cv2.waitKey(1)

            if center is None:
                raise RuntimeError("No ArUco marker detected during calibration. Make sure it is visible to the camera.")

            image_points.append(center)
            robot_xy_points.append(np_module.array([x, y], dtype=np_module.float32))
            print(f"Captured image point {center} for robot XY ({x}, {y})")

        return image_points, robot_xy_points

    @staticmethod
    def compute_homography(image_points: Sequence[np.ndarray], robot_points: Sequence[np.ndarray]) -> HomographyResult:
        img_pts = np.array(image_points, dtype=np.float32)
        rob_pts = np.array(robot_points, dtype=np.float32)
        if img_pts.shape[0] < 4:
            raise ValueError("At least 4 points are required to compute a homography")

        H, mask = cv2.findHomography(img_pts, rob_pts, method=cv2.RANSAC)
        if H is None:
            raise RuntimeError("Homography estimation failed")
        inliers = int(mask.sum()) if mask is not None else img_pts.shape[0]
        print(f"Homography computed with {inliers}/{len(img_pts)} inliers")
        return HomographyResult(homography=H, inverse=np.linalg.inv(H))

    def pixel_to_robot(self, pixel: Sequence[float], homography: np.ndarray) -> Tuple[float, float]:
        pt = np.array([[pixel[0], pixel[1], 1.0]], dtype=np.float64).T
        mapped = homography @ pt
        mapped /= mapped[2]
        return float(mapped[0]), float(mapped[1])

    def follow_marker(
        self,
        robot: FreenoveArmClient,
        homography: HomographyResult,
        z_height: float,
        move_speed: int = 50,
    ) -> None:
        print("Starting follow mode. Press 'q' to exit.")
        while True:
            frame = self.read_frame()
            vis, center = self.detect_marker(frame)
            status = "Marker not found"

            if center is not None:
                x, y = self.pixel_to_robot(center, homography.homography)
                status = f"Target -> X: {x:.1f} mm, Y: {y:.1f} mm, Z: {z_height:.1f} mm"
                robot.move_to(x, y, z_height, speed=move_speed)
                self.cv2.circle(vis, (int(center[0]), int(center[1])), 10, (255, 0, 0), 2)
            self.cv2.putText(
                vis, status, (10, 30), self.cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
            self.cv2.imshow("follow", vis)
            key = self.cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        self.camera.release()
        self.cv2.destroyAllWindows()


def default_calibration_points(z_height: float) -> List[Tuple[float, float, float]]:
    grid = [
        (80, 120, z_height),
        (80, 200, z_height),
        (80, 280, z_height),
        (160, 120, z_height),
        (160, 200, z_height),
        (160, 280, z_height),
        (240, 120, z_height),
        (240, 200, z_height),
        (240, 280, z_height),
    ]
    return grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Planar hand-eye calibration for Freenove arm + USB camera")
    parser.add_argument("--mode", choices=["calibrate", "follow"], required=True, help="Workflow to run")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Camera source (index like '0', RTSP/HTTP URL, or virtual cam path).",
    )
    parser.add_argument("--cam-width", type=int, default=1280, help="Requested camera width (pixels)")
    parser.add_argument("--cam-height", type=int, default=720, help="Requested camera height (pixels)")
    parser.add_argument("--cam-fps", type=int, default=30, help="Requested camera FPS")
    parser.add_argument("--z-height", type=float, default=70.0, help="Fixed Z height used for calibration and following (mm)")
    parser.add_argument("--host", type=str, default="10.149.65.232", help="Robot IP address")
    parser.add_argument("--port", type=int, default=5000, help="Robot TCP port (matches client.py default)")
    parser.add_argument("--dry-run", action="store_true", help="Print robot commands instead of sending them")
    parser.add_argument("--skip-enable", action="store_true", help="Do not send the S8 motor enable command on connect")
    parser.add_argument("--speed", type=int, default=50, help="Robot move speed hint (not all firmware uses this)")
    parser.add_argument("--settle", type=float, default=1.0, help="Delay after each move during calibration (s)")
    parser.add_argument("--step", action="store_true", help="Pause for Enter before each calibration move")
    parser.add_argument(
        "--home-first",
        dest="home_first",
        action="store_true",
        default=True,
        help="Send S10 F1 right after enabling motors (default: on; use --no-home-first to skip)",
    )
    parser.add_argument(
        "--no-home-first",
        dest="home_first",
        action="store_false",
        help="Skip the S10 F1 homing step",
    )
    parser.add_argument("--ground-clearance", type=float, help="Send S3 to set the ground clearance height (mm)")
    parser.add_argument("--verbose", action="store_true", help="Print every command sent to the arm")
    parser.add_argument(
        "--aruco-dict",
        type=str,
        default=DEFAULT_ARUCO_NAME,
        choices=["auto"] + SUPPORTED_ARUCO_NAMES,
        help='Marker dictionary name, or "auto" to scan all supported ArUco/AprilTag sets.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    camera_cfg = CameraConfig(source=args.source, width=args.cam_width, height=args.cam_height, fps=args.cam_fps)
    calibrator = PlaneCalibrator(camera_cfg, dictionary_name=args.aruco_dict)

    with FreenoveArmClient(
        host=args.host,
        port=args.port,
        dry_run=args.dry_run,
        auto_enable=not args.skip_enable,
        verbose=args.verbose,
    ) as arm:
        # Mirror quick_move / official UI: home immediately after enable.
        if args.home_first:
            arm.return_to_sensor_point(1)
            arm.wait(0.5)

        if args.ground_clearance is not None:
            arm.set_ground_clearance(args.ground_clearance)
            arm.wait(0.1)

        if args.mode == "calibrate":
            robot_points = default_calibration_points(args.z_height)
            img_pts, rob_pts = calibrator.collect_correspondences(
                robot_points, arm, settle_time=args.settle, step_mode=args.step
            )
            H = calibrator.compute_homography(img_pts, rob_pts)
            H.save()
            print("Homography saved to save_parms/homography.npy and save_parms/homography_inv.npy")
        elif args.mode == "follow":
            H = HomographyResult.load()
            calibrator.follow_marker(arm, H, z_height=args.z_height, move_speed=args.speed)


if __name__ == "__main__":
    main()

"""
Planar hand–eye calibration script for the Freenove robot arm.

This version speaks the same TCP text protocol as the Freenove desktop app,
via :class:`freenove_arm.FreenoveArmClient`.  It estimates a 2D homography
between camera pixels and the robot table plane using an ArUco marker attached
to the gripper.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import cv2.aruco as aruco
import numpy as np

from freenove_arm import FreenoveArmClient

# Default ArUco dictionary and marker size (in meters, only used for drawing)
ARUCO_DICT = aruco.DICT_5X5_100
MARKER_LENGTH_METERS = 0.02


@dataclass
class CameraConfig:
    camera_id: int = 0
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
    def __init__(self, camera_cfg: CameraConfig, dictionary_id: int = ARUCO_DICT):
        self.camera_cfg = camera_cfg
        self.dictionary = aruco.getPredefinedDictionary(dictionary_id)
        self.parameters = aruco.DetectorParameters()
        self.camera = self._open_camera(camera_cfg)

    def _open_camera(self, cfg: CameraConfig) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(cfg.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        if not cap.isOpened():
            raise RuntimeError("Could not open USB camera")
        return cap

    def read_frame(self) -> np.ndarray:
        ok, frame = self.camera.read()
        if not ok:
            raise RuntimeError("Failed to read from camera")
        return frame

    def detect_marker(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)

        if ids is None or len(ids) == 0:
            return frame, None

        aruco.drawDetectedMarkers(frame, corners)
        focal_length = max(self.camera_cfg.width, self.camera_cfg.height)
        camera_matrix = np.array(
            [[focal_length, 0, self.camera_cfg.width / 2], [0, focal_length, self.camera_cfg.height / 2], [0, 0, 1]],
            dtype=np.float32,
        )
        dist_coeffs = np.zeros((1, 5))
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH_METERS, camera_matrix, dist_coeffs)
        for rvec, tvec in zip(rvecs, tvecs):
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH_METERS)

        corner = corners[0][0]
        center = corner.mean(axis=0)
        cv2.circle(frame, tuple(center.astype(int)), 5, (0, 0, 255), -1)
        cv2.putText(frame, "Aruco center", (int(center[0]) + 5, int(center[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame, center

    def collect_correspondences(
        self,
        robot_points: Sequence[Tuple[float, float, float]],
        robot: FreenoveArmClient,
        settle_time: float = 1.0,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        image_points: List[np.ndarray] = []
        robot_xy_points: List[np.ndarray] = []

        for idx, (x, y, z) in enumerate(robot_points):
            print(f"Moving to calibration point {idx + 1}/{len(robot_points)}: ({x}, {y}, {z})")
            robot.move_to(x, y, z)
            robot.wait(settle_time)

            frame = self.read_frame()
            vis, center = self.detect_marker(frame)
            cv2.imshow("calibration", vis)
            cv2.waitKey(1)

            if center is None:
                raise RuntimeError("No ArUco marker detected during calibration. Make sure it is visible to the camera.")

            image_points.append(center)
            robot_xy_points.append(np.array([x, y], dtype=np.float32))
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
                cv2.circle(vis, (int(center[0]), int(center[1])), 10, (255, 0, 0), 2)
            cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("follow", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        self.camera.release()
        cv2.destroyAllWindows()


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
    parser.add_argument("--camera-id", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--z-height", type=float, default=70.0, help="Fixed Z height used for calibration and following (mm)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Robot IP address")
    parser.add_argument("--port", type=int, default=5000, help="Robot TCP port (matches client.py default)")
    parser.add_argument("--dry-run", action="store_true", help="Print robot commands instead of sending them")
    parser.add_argument("--skip-enable", action="store_true", help="Do not send the S8 motor enable command on connect")
    parser.add_argument("--speed", type=int, default=50, help="Robot move speed hint (not all firmware uses this)")
    parser.add_argument("--settle", type=float, default=1.0, help="Delay after each move during calibration (s)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    camera_cfg = CameraConfig(camera_id=args.camera_id)
    calibrator = PlaneCalibrator(camera_cfg)

    with FreenoveArmClient(
        host=args.host,
        port=args.port,
        dry_run=args.dry_run,
        auto_enable=not args.skip_enable,
    ) as arm:
        if args.mode == "calibrate":
            robot_points = default_calibration_points(args.z_height)
            img_pts, rob_pts = calibrator.collect_correspondences(robot_points, arm, settle_time=args.settle)
            H = calibrator.compute_homography(img_pts, rob_pts)
            H.save()
            print("Homography saved to save_parms/homography.npy and save_parms/homography_inv.npy")
        elif args.mode == "follow":
            H = HomographyResult.load()
            calibrator.follow_marker(arm, H, z_height=args.z_height, move_speed=args.speed)


if __name__ == "__main__":
    main()

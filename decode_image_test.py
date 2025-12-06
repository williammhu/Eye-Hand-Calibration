"""
Decode a single image by estimating barcode orientation and rotating accordingly.
Run with:
    python decode_image_test.py --path <image_path>
"""

import argparse
import sys
from typing import Iterable

import cv2
import numpy as np
from pyzbar import pyzbar

# Default test image path; override via CLI.
IMAGE_PATH = r"D:\Eye-Hand-Calibration\1.png"


def rotate_bound(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an image while keeping the full content in frame."""

    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH))


def barcode_angle(gray: np.ndarray) -> float:
    """Estimate barcode angle from a grayscale image."""

    ret, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((8, 8), np.uint8)
    dilation = cv2.dilate(binary, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)
    erosion = cv2.erode(erosion, kernel, iterations=1)
    erosion = cv2.erode(erosion, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(
        erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if len(contours) == 0:
        rect = [0, 0, 0]
    else:
        rect = cv2.minAreaRect(contours[0])
    return float(rect[2])


def bar(image: np.ndarray, angle: float) -> Iterable[pyzbar.Decoded]:
    """Rotate image to upright position and decode with pyzbar."""

    rotated = rotate_bound(image, 0 - angle)
    roi = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
    return pyzbar.decode(roi)


def decode_image(img_path: str, debug: bool = False) -> bool:
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image: {img_path}")
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    angle = barcode_angle(gray)
    if debug:
        print(f"Estimated angle: {angle:.2f} degrees")

    decoded = bar(img, angle)
    if not decoded:
        print("No codes found.")
        return False

    for obj in decoded:
        try:
            text = obj.data.decode("utf-8", errors="ignore")
        except Exception:
            text = "<decode error>"
        print(f"Type={obj.type} | Data={text} | Rect={obj.rect}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Decode a single image by estimating barcode orientation."
    )
    parser.add_argument("--path", default=IMAGE_PATH, help="Image path (default: IMAGE_PATH in file)")
    parser.add_argument("--debug", action="store_true", help="Print estimated angle")
    args = parser.parse_args()

    ok = decode_image(args.path, debug=args.debug)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

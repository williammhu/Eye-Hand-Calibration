"""
Simple pyzbar image decoder for debugging.
Set IMAGE_PATH to your test image and run:
    python decode_image_test.py
Use --path <img> to override via CLI.
"""

import argparse
import sys
from typing import List, Tuple

import cv2
import numpy as np
from pyzbar import pyzbar

# Hardcoded test image path; change this or pass --path at runtime.
IMAGE_PATH = r"D:\Eye-Hand-Calibration\1.png"


def _build_symbol_list(allow_pdf417: bool):
    if allow_pdf417:
        return None  # use all built-in types
    return [
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


def _preprocess_variants(img: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    Generate several enhanced grayscale versions of the image to boost decode rate.
    The list order goes from least to most aggressive to avoid over-processing.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Upscale a bit to give the decoder more pixels to work with.
    up = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

    # Contrast Limited Adaptive Histogram Equalization for uneven lighting.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(up)

    # Gentle blur to suppress sensor noise.
    blur = cv2.GaussianBlur(clahe, (3, 3), 0)

    # Binary versions (two styles) plus a morphology-cleaned version.
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=1)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)

    return [
        ("gray_up", up),
        ("clahe", clahe),
        ("blur", blur),
        ("otsu", otsu),
        ("adaptive", adaptive),
        ("morph", morph),
    ]


def decode_image(img_path: str, allow_pdf417: bool = False, debug: bool = False) -> bool:
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image: {img_path}")
        return False

    symbols = _build_symbol_list(allow_pdf417)
    variants = _preprocess_variants(img)

    seen = set()
    found_any = False

    for name, variant in variants:
        decoded = pyzbar.decode(variant, symbols=symbols)
        if debug:
            print(f"[{name}] decoded {len(decoded)} symbol(s)")
        for obj in decoded:
            key = (obj.type, obj.data)
            if key in seen:
                continue
            seen.add(key)
            found_any = True
            try:
                text = obj.data.decode("utf-8", errors="ignore")
            except Exception:
                text = "<decode error>"
            print(f"Variant={name} | Type={obj.type} | Data={text} | Rect={obj.rect}")

        # Early exit if we already got something on a gentle transform.
        if found_any and name in {"gray_up", "clahe", "blur"}:
            break

    if not found_any:
        print("No codes found.")
    return found_any


def main():
    parser = argparse.ArgumentParser(description="Decode a single image with pyzbar (with preprocessing).")
    parser.add_argument("--path", default=IMAGE_PATH, help="Image path (default: IMAGE_PATH in file)")
    parser.add_argument("--allow-pdf417", action="store_true", help="Enable PDF417 decoding")
    parser.add_argument("--debug", action="store_true", help="Print decode counts for each preprocessing step")
    args = parser.parse_args()

    ok = decode_image(args.path, allow_pdf417=args.allow_pdf417, debug=args.debug)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

"""
简单用 zxing 直接解码一张图片（无任何预处理/旋转）。
用法:
    python decode_image_test.py --path <image_path>
"""

import argparse
import os
import sys

import zxing

# 默认测试图片路径，可用 CLI 参数覆盖
IMAGE_PATH = r"D:\Eye-Hand-Calibration\1.png"


def decode_image(img_path: str, debug: bool = False) -> bool:
    if not os.path.isfile(img_path):
        print(f"文件不存在: {img_path}")
        return False

    reader = zxing.BarCodeReader()
    # try_harder=True 可在弱信号时多尝试几次
    result = reader.decode(img_path, try_harder=True)

    if debug:
        print("reader:", reader)
        print("result object:", result)

    if result is None:
        print("未识别到条码/二维码。")
        return False

    print(f"Format={getattr(result, 'format', None)} | Type={getattr(result, 'type', None)}")
    print(f"Parsed={getattr(result, 'parsed', None)}")
    print(f"Raw={getattr(result, 'raw', None)}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Decode image using zxing (no preprocessing).")
    parser.add_argument("--path", default=IMAGE_PATH, help="图片路径（默认: 文件内的 IMAGE_PATH）")
    parser.add_argument("--debug", action="store_true", help="打印调试信息")
    args = parser.parse_args()

    ok = decode_image(args.path, debug=args.debug)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

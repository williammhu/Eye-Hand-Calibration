"""
Minimal single-move tester for the Freenove robot arm.

Usage example (real move):
    python quick_move.py --host 10.149.65.232 --x 160 --y 200 --z 60 --verbose

Dry run (just print commands):
    python quick_move.py --dry-run --verbose
"""

from __future__ import annotations

import argparse
import time

from freenove_arm import FreenoveArmClient


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Send one move command to the Freenove arm (mirrors official UI flow).")
    p.add_argument("--host", type=str, default="10.149.65.232", help="Robot IP address")
    p.add_argument("--port", type=int, default=5000, help="Robot TCP port")
    p.add_argument("--x", type=float, default=160.0, help="Target X (mm)")
    p.add_argument("--y", type=float, default=200.0, help="Target Y (mm)")
    p.add_argument("--z", type=float, default=70.0, help="Target Z (mm)")
    p.add_argument("--dwell-ms", type=int, help="Optional G4 pause (ms) after the move")
    p.add_argument("--settle", type=float, default=1.0, help="Sleep time after move (seconds)")
    p.add_argument(
        "--home-first",
        dest="home_first",
        action="store_true",
        default=True,
        help="Send S10 F1 (home) right after enabling motors (default: on, use --no-home-first to skip)",
    )
    p.add_argument(
        "--no-home-first",
        dest="home_first",
        action="store_false",
        help="Skip the S10 F1 homing step",
    )
    p.add_argument("--skip-enable", action="store_true", help="Skip sending S8 enable")
    p.add_argument("--dry-run", action="store_true", help="Print commands instead of sending")
    p.add_argument("--verbose", action="store_true", help="Print every command sent")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with FreenoveArmClient(
        host=args.host,
        port=args.port,
        dry_run=args.dry_run,
        auto_enable=not args.skip_enable,
        verbose=args.verbose,
    ) as arm:
        # The official UI sends S10 F1 immediately after S8 E0 (first move after load).
        if args.home_first:
            arm.return_to_sensor_point(1)
            arm.wait(0.5)

        print(f"Moving to X={args.x:.1f}, Y={args.y:.1f}, Z={args.z:.1f}")
        arm.move_to(args.x, args.y, args.z, dwell_ms=args.dwell_ms)
        arm.wait(args.settle)

    print("Done.")


if __name__ == "__main__":
    main()

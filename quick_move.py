"""
Minimal single-move tester for the Freenove robot arm.

Usage example (real move):
 python quick_move.py --host 10.149.65.232 --interactive --verbose

Dry run (just print commands):
    python quick_move.py --dry-run --verbose
    100 200 90
     0 200 90
    -100 200 90
     0 150 90
     0 250 90


"""

from __future__ import annotations

import argparse
import time

from freenove_arm import FreenoveArmClient


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Send one move command to the Freenove arm (mirrors official UI flow).")
    p.add_argument("--host", type=str, default="10.149.65.232", help="Robot IP address")
    p.add_argument("--port", type=int, default=5000, help="Robot TCP port")
    p.add_argument("--x", type=float, default=-170.0, help="Target X (mm)")
    p.add_argument("--y", type=float, default=160.0, help="Target Y (mm)")
    p.add_argument("--z", type=float, default=100.0, help="Target Z (mm)")
    p.add_argument("--dwell-ms", type=int, help="Optional G4 pause (ms) after the move")
    p.add_argument("--settle", type=float, default=1.0, help="Sleep time after move (seconds)")
    p.add_argument(
        "--interactive",
        action="store_true",
        help="Keep the motors enabled and accept repeated moves until you quit",
    )
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

        if args.interactive:
            print("Interactive mode: enter 'x y z [dwell_ms]' per line, or 'q' to quit.")
            while True:
                try:
                    line = input("> ").strip()
                except (KeyboardInterrupt, EOFError):
                    print("\nExiting interactive mode.")
                    break

                if not line:
                    continue
                if line.lower() in {"q", "quit", "exit"}:
                    break

                parts = line.split()
                if len(parts) < 3:
                    print("Please enter at least x y z (mm). Optional fourth value is dwell_ms.")
                    continue

                try:
                    x, y, z = map(float, parts[:3])
                    dwell_ms = int(parts[3]) if len(parts) >= 4 else args.dwell_ms
                except ValueError:
                    print("Could not parse numbers; try again.")
                    continue

                print(f"Moving to X={x:.1f}, Y={y:.1f}, Z={z:.1f}")
                arm.move_to(x, y, z, dwell_ms=dwell_ms)
                arm.wait(args.settle)
        else:
            print(f"Moving to X={args.x:.1f}, Y={args.y:.1f}, Z={args.z:.1f}")
            arm.move_to(args.x, args.y, args.z, dwell_ms=args.dwell_ms)
            arm.wait(args.settle)

    print("Done.")


if __name__ == "__main__":
    main()

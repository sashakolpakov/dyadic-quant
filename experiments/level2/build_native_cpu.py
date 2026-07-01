from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dyadic_quant.level2.native import build_native_cpu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Level 2 native CPU kernels.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_native_cpu(force=args.force)
    print(output)


if __name__ == "__main__":
    main()

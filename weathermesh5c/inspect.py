from __future__ import annotations

import argparse
from pathlib import Path

from .assets import RuntimeBundle
from .checkpoint import checkpoint_info, validate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect WeatherMesh-5c runtime inputs")
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--verify-checksum", action="store_true")
    parser.add_argument("--inspect-tensors", action="store_true")
    args = parser.parse_args()

    bundle = RuntimeBundle(args.bundle_root)
    bundle.validate(checksums=args.verify_checksum)
    validate_checkpoint(bundle.checkpoint, checksum=False)
    info = checkpoint_info(
        bundle.checkpoint,
        verify_checksum=False,
        inspect_tensors=args.inspect_tensors,
    )
    print(f"bundle={bundle.root}")
    print(f"checkpoint={info.path}")
    print(f"size_bytes={info.size_bytes}")
    if info.keys is not None:
        print(f"keys={info.keys}")
        print(f"tensor_elements={info.tensor_elements}")
    print("runtime_bundle=ok")


if __name__ == "__main__":
    main()

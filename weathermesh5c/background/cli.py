from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .download import (
    download_oisst,
    download_open_ifs,
    existing_oisst_path,
    latest_open_cycle,
    open_ifs_path,
)
from .prepare import prepare_background


def _cycle(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Cycle must use YYYYMMDDHH in UTC") from error


def _add_common(parser: argparse.ArgumentParser, *, cycle_required: bool) -> None:
    parser.add_argument("--cycle", type=_cycle, required=cycle_required)
    parser.add_argument("--member", type=int, default=1)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))


def main() -> None:
    parser = argparse.ArgumentParser(description="WeatherMesh-5c Open IFS background ingestion")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("fetch", help="Download Open IFS and NOAA OISST")
    _add_common(fetch, cycle_required=False)
    fetch.add_argument("--workers", type=int, default=8)

    process = subparsers.add_parser("process", help="Prepare a downloaded 5C background")
    _add_common(process, cycle_required=True)
    process.add_argument("--processed-root", type=Path, default=Path("data/backgrounds"))

    ingest = subparsers.add_parser("ingest", help="Download and prepare one 5C background")
    _add_common(ingest, cycle_required=False)
    ingest.add_argument("--processed-root", type=Path, default=Path("data/backgrounds"))
    ingest.add_argument("--workers", type=int, default=8)

    args = parser.parse_args()
    cycle = args.cycle or latest_open_cycle()
    print(f"cycle={cycle:%Y%m%d%H}Z member={args.member}")

    if args.command in {"fetch", "ingest"}:
        grib = download_open_ifs(
            args.raw_root,
            cycle,
            member=args.member,
            workers=args.workers,
        )
        oisst = download_oisst(args.raw_root, cycle)
        print(f"open_ifs={grib}")
        print(f"oisst={oisst}")
    else:
        grib = open_ifs_path(args.raw_root, cycle, args.member)
        if not grib.is_file():
            raise FileNotFoundError(f"Open IFS GRIB is missing: {grib}")
        oisst = existing_oisst_path(args.raw_root, cycle)

    if args.command in {"process", "ingest"}:
        output = prepare_background(
            grib,
            oisst,
            args.processed_root,
            cycle=cycle,
            member=args.member,
        )
        print(f"background={output}")
        print(f"aida_valid_time={cycle + timedelta(hours=3):%Y%m%d%H}Z")


if __name__ == "__main__":
    main()

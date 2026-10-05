from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .download import download, latest_common_cycle
from .process import process_file
from .sources import PUBLIC_SOURCES, SOURCES
from .storage import write_hours
from .windborne import fetch_windborne


def _cycle(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Cycle must use YYYYMMDDHH in UTC") from error


def _source_list(values: list[str]) -> tuple[str, ...]:
    if values == ["all"]:
        return PUBLIC_SOURCES
    unknown = set(values) - set(PUBLIC_SOURCES)
    if unknown:
        raise ValueError(f"Unknown NOAA observation sources: {', '.join(sorted(unknown))}")
    return tuple(values)


def _print_hours(source: str, hours) -> None:
    observations = sum(len(payload["x"]) for payload in hours.values())
    print(f"source={source} hours={len(hours)} observations={observations}")


def _process(args, source: str, path: Path) -> None:
    hours = process_file(source, path)
    written = write_hours(args.processed_root, source, hours)
    _print_hours(source, hours)
    for output in written:
        print(f"written={output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="WeatherMesh-5c observation ingestion")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("fetch", help="Download public NOAA observation BUFR")
    fetch.add_argument("sources", nargs="+", help="NOAA observation source names, or 'all'")
    fetch.add_argument("--cycle", type=_cycle)
    fetch.add_argument("--feed", choices=("gdas", "gfs", "rap"), default="gdas")
    fetch.add_argument("--raw-root", type=Path, default=Path("data/raw"))

    process = subparsers.add_parser("process", help="Process one downloaded source file")
    process.add_argument("source", choices=tuple(SOURCES))
    process.add_argument("path", type=Path)
    process.add_argument("--processed-root", type=Path, default=Path("data/processed"))

    ingest = subparsers.add_parser("ingest", help="Download and process one observation cycle")
    ingest.add_argument("sources", nargs="+", help="NOAA observation source names, or 'all'")
    ingest.add_argument("--cycle", type=_cycle)
    ingest.add_argument("--feed", choices=("gdas", "gfs", "rap"), default="gdas")
    ingest.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    ingest.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    ingest.add_argument(
        "--windborne",
        action="store_true",
        help="Also ingest official WindBorne super-observations for AIDA",
    )

    args = parser.parse_args()
    if args.command == "process":
        _process(args, args.source, args.path)
        return

    sources = _source_list(args.sources)
    cycle = args.cycle or latest_common_cycle(sources, feed=args.feed)
    print(f"cycle={cycle:%Y%m%d%H} feed={args.feed}")
    for source in sources:
        path = download(source, args.raw_root, cycle=cycle, feed=args.feed)
        print(f"downloaded={path}")
        if args.command == "ingest":
            _process(args, source, path)
    if args.command == "ingest" and args.windborne:
        valid_time = cycle + timedelta(hours=3)
        hours = fetch_windborne(valid_time)
        written = write_hours(args.processed_root, "windborne", hours)
        _print_hours("windborne", hours)
        for output in written:
            print(f"written={output}")


if __name__ == "__main__":
    main()

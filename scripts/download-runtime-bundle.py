from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote

import requests

from weathermesh5c import RuntimeBundle

HF_REVISION = "260377d3534a098a5a3369adab309f897a351bc3"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the public WeatherMesh-5c runtime bundle")
    parser.add_argument("--output", type=Path, default=Path("data/runtime-bundle"))
    parser.add_argument("--revision", default=HF_REVISION)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base_url = (
        "https://huggingface.co/windborne/weathermesh-5c/resolve/"
        f"{quote(args.revision, safe='')}"
    )

    with TemporaryDirectory(prefix=".download-", dir=args.output) as temporary:
        with requests.Session() as session:
            def resolve(relative: str) -> Path:
                path = Path(temporary) / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                print(f"downloading={relative}", flush=True)
                with session.get(
                    f"{base_url}/{relative}", stream=True, timeout=(10, 300)
                ) as response:
                    response.raise_for_status()
                    with path.open("wb") as output:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            output.write(chunk)
                return path

            bundle = RuntimeBundle.materialize(resolve, args.output, hardlinks=True)
    print(f"runtime_bundle={bundle.root}")
    print("checksums=ok")


if __name__ == "__main__":
    main()

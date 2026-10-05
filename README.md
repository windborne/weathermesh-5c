# WeatherMesh-5c and AIDA

This repository contains the standalone inference runtime for WeatherMesh-5c and its
AIDA analysis system.

The model definition comes from WindBorne `deep` commit
`5f41011701988aa84f05279041a6eb2a0b3a3783`. The checkpoint is the production
WeatherMesh-5c `2b81` state dictionary.

## Current scope

The runtime provides:

- a focused `build_model()` function for WeatherMesh-5c and AIDA;
- strict, safe state-dictionary loading;
- checkpoint and runtime-asset validation;
- ECMWF Open IFS ensemble download and 5C background preparation;
- NOAA OISST download and sea-surface-temperature preparation;
- NOAA NOMADS download and processing for AMSU-A, MHS, ATMS, IASI, GPSRO,
  radiosonde, satellite-wind, and surface observations;
- official WindBorne API ingestion for super-observations;
- exact 5C observation normalization and model-batch assembly;
- a portable, checksummed runtime bundle with safe numeric asset formats;
- atomic materialization of local or remote runtime packages into a portable cache;
- one `wm5c-run` command for raw public inputs through the operational ensemble products;
- a single-GPU AIDA, forecast-latent, and global-decoder smoke test;
- an inference-only model core, with no training or evaluation entry points.

The companion Earth2Studio integration imports this package as its model-specific
runtime instead of duplicating the inference implementation.

The package does not contain training loops, losses, optimizer helpers, checkpoint
assembly, plotting, internal server control, credentials, or cloud-control code.

## Environment

WeatherMesh-5c uses a repository-local `.venv`. Run project commands through the
launcher so an inherited environment setting cannot select a shared or unrelated
Python environment:

```bash
./scripts/wm5c-uv.sh sync
```

The launcher resolves the checkout location and forces `<checkout>/.venv`. Sync and
run commands use the committed CUDA-12.8 lock without changing it. The lock command
selects the PyTorch CUDA-12.8 index without putting that index in distributed package
metadata. Builds use the synchronized environment without installing isolated build
dependencies. The launcher refuses symlinked environments and arguments that could
select another project or environment. Do not call `uv` directly from this repository.

## Runtime bundle

The checkpoint and static runtime assets are hosted separately on
[Hugging Face](https://huggingface.co/windborne/weathermesh-5c). A local copy keeps
the same directory layout:

```text
runtime-bundle/
├── manifest.json
├── weights/weather-mesh-5c-aida.pt
└── assets/global/
    ├── land_mask.npy
    ├── soil_type.npy
    ├── topography.npy
    └── meshes/
```

The manifest records the role, size, and SHA-256 of every file. The schema-version 3
reference bundle has 10 files and is 3,535,775,581 bytes. Solar elevation and direct
radiation are deterministic temporal features. The runtime generates each required global
field from the original production formula and keeps a bounded in-memory cache instead of
downloading a 36.5 GB climatology.

The bundle is about 3.54 GB (3.29 GiB). Allow at least 8 GB for download and
materialization when the two caches cannot share files. Forecast output storage is
separate and depends on the member count, lead schedule, and product selection.

Earth2Studio can resolve these files from NGC, Hugging Face, S3, or a local directory. It
materializes the manifest into its model cache, preserves the bundle directory structure,
reuses valid files, and verifies all checksums before model loading. On one file system,
the download cache and runtime layout share immutable bytes through hard links. The
materializer falls back to a reflink or copy when a hard link is unavailable.

Build a bundle from authorized production inputs:

```shell
./scripts/wm5c-uv.sh run --no-sync python scripts/build-runtime-bundle.py \
  --checkpoint /path/to/weather-mesh-5c-aida.pt \
  --consts /path/to/global-assets \
  --output /path/to/runtime-bundle
```

The builder converts the trusted legacy mesh pickles into `.npz` files. Inference never
opens a pickle. The builder is the only pickle reader and must receive trusted source
files. Its normal mode copies all fixed inputs. `--link-existing-files` uses hard links
where possible and is only for local validation.

Run the complete integrity check after transfer:

```bash
./scripts/wm5c-uv.sh run --no-sync python -m weathermesh5c.inspect \
  --bundle-root /path/to/runtime-bundle \
  --verify-checksum \
  --inspect-tensors
```

The checkpoint contract remains:

```text
size:       2841543169 bytes
sha256:     20d6a06cf75f6fa42c8796bc39fa7e426ea08be390a895b6236a6f03aabe73d2
keys:       984
elements:   762997823
```

## Load the model

```python
from weathermesh5c import load_model

model = load_model("/path/to/runtime-bundle", device="cuda:0")
```

The loader uses `torch.load(..., weights_only=True)` and strict key matching. It does not
load training configuration or executable pickle objects.

The model input code preserves one trained radiation-feature convention: UTC hour selects
the generated source day, and the source hour is zero. This convention looks unusual, but
changing it changes the checkpoint's numerical input and forecast.

## Earth2Studio integration

The Earth2Studio integration provides `WeatherMesh5C` for forecasts from weather
fields, `AIDA` for analysis from background fields and observations, and
`AIDAWeatherMesh5C` for forecasts initialized by AIDA. All three use this runtime.

## End-to-end forecast

After the observation decoders are installed, one command selects a compatible cycle,
downloads and prepares the Open IFS backgrounds and all public NOAA observations, runs
AIDA and WeatherMesh-5c, and writes physical ensemble forecasts plus a JSON manifest:

```bash
./scripts/wm5c-uv.sh run wm5c-run \
  --bundle-root /path/to/runtime-bundle
```

Add `--windborne` and set `WB_API_KEY` to include official WindBorne
super-observations in AIDA.

The default profile matches the operational realtime inference calculation:

- Open IFS perturbed members 1 through 24 each receive a separate AIDA analysis;
- each latent follows one path through hour 69;
- the learned forker creates eight paths at hour 72;
- gridded surface products run every three hours through hour 360;
- pressure products run at hours 0, 6, …, 120 after a two-update analysis, or
  hours 3, 9, …, 117 after a three-update analysis;
- overall mean and sample standard deviation use all 24 paths before the fork and all
  192 paths at and after it.

The command processes one member and one learned branch at a time. It follows the
realtime producer and aggregator design within that portable, single-GPU limit:

- prepared backgrounds are loaded one member at a time;
- processed observations are loaded and normalized once, then copied for each AIDA pass;
- only compact float16 decoder products cross from GPU to CPU;
- a bounded pinned-memory queue overlaps CUDA copies and CPU aggregation;
- member and pressure-supermember stores keep means only; and
- bounded worker processes write independent forecast files in parallel.

Disk-backed online statistics bound forecast aggregation memory. Observation volume
also affects GPU memory; an uncapped run is not guaranteed to fit every GPU. The complete
operational profile is still a large forecast. Budget at least 250 GB of local space
when member products are enabled. It also needs substantial runtime on one GPU.

Each top-level `forecast-fNNN.zarr` contains ensemble mean and standard deviation.
`members/memberNN` contains the operational gridded surface member products. After hour
72, each published member is the mean of its eight learned branches.
`pressure-supermembers` contains one pressure member for each group of six Open IFS
members, as in the realtime runner. The manifest records the exact members, fork
schedule, random seed, observations, checkpoint, runtime, and trajectory count.

Lead hours and filenames are relative to the final AIDA analysis. Forecast files also
record the operational `forecast_reference_time` (the preceding six-hour UTC boundary)
and `forecast_hour`. For a three-update analysis, that forecast hour is three hours
larger than the analysis-relative lead. The physical valid time is unchanged.

Use separate `--background-root` and `--observations-root` directories for each
preprocessing version. Do not mix prepared backgrounds or processed GPSRO records
from different versions. The raw download cache can be reused.

Use `--cycle YYYYMMDDHH` for a fixed cycle. Leads are relative to the AIDA valid time,
which is six or nine hours after the Open IFS cycle. AIDA advances and assimilates
observations every three hours before starting the forecast. Fixed-cycle and prepared-
background runs default to two updates (cycle +3 and +6 hours); use
`--assimilation-steps 3` for updates at +3, +6, and +9 hours. Automatic live runs use
three updates when the selected background is more than 7.75 hours old, matching the
production timing rule. `--assimilation-steps 2` or `3` overrides that choice.
Every update must have observations; missing windows fail rather than being skipped.
Automatic selection requires NOAA files for all update windows, so public-feed
publication delays can select an older background than the internal production feed.
The manifest records the update times and observation counts for each window.
For a bounded validation run, replace the operational schedule explicitly:

```bash
./scripts/wm5c-uv.sh run --no-sync wm5c-run \
  --bundle-root /path/to/runtime-bundle \
  --background /path/to/prepared-background.npz \
  --members 1 \
  --lead-hours 12 \
  --pressure-lead-hours 12 \
  --fork-hour 12 \
  --fork-count 2 \
  --skip-observation-ingest
```

Observations are uncapped by default, matching production. For smaller test runs,
use `--max-observations 20000`. This selects rows without replacement using Deep's
`torch.randperm` method, separately for each source and assimilation window. The
runner uses `--seed` for repeatability and reuses each subset across ensemble members.
Uncapped runs may need more memory; there is no automatic data reduction. The command
checks for at least 16 GiB of free GPU memory before model loading.
Use `--skip-member-output` to write only overall mean
and standard deviation. `--output-workers` controls output workers and
defaults to four. Reduce it on systems with limited CPU memory or slow storage. Existing
forecast directories are not replaced unless `--overwrite` is present. That option
removes the exact cycle output directory before the new run so stale member files cannot
survive.

### Output formats

Compressed Zarr is the default in the CLI, Python, and Earth2Studio workflows. It
writes Deep-style Zarr v2 ZIP stores with `.zarr` filenames. Select
`--output-format netcdf`, or `ForecastRequest(output_format="netcdf")`, for
compressed NetCDF files with `.nc` filenames. Both formats retain the same forecast
values, times, ensemble statistics, and field metadata.

Zarr packs surface and pressure variables into separate arrays. It uses all-variable
chunks with 45×45 spatial cells and Blosc/Zstd level 1 with byte shuffle, as in Deep.
The writer does not apply Deep's final value rounding. Pressure variables are ordered
by field, then level. Each packed array's `variable_metadata` attribute records field
names, units, and pressure levels. Zarr does not claim the NetCDF writer's CF data-variable
conformance, because one packed array contains fields with different units.

Read a Zarr file with an explicit ZIP store:

```python
import xarray as xr
from zarr.storage import ZipStore

with ZipStore("forecast-f012.zarr", mode="r") as store:
    with xr.open_zarr(store, consolidated=False) as dataset:
        temperature = dataset.surface.sel(surface_variable="air_temperature_2m").load()
        pressure_temperature = dataset.pressure.sel(pressure_variable="air_temperature_500").load()
```

`--no-output-compression`, or `ForecastRequest(output_compression=False)`, disables
compression without changing values. Use `--output-format netcdf --no-output-compression`
for uncompressed NetCDF, or
`ForecastRequest(output_format="netcdf", output_compression=False)` in Python.
Each run writes only the selected format.

## Open IFS background ingestion

Download the latest available 0.25-degree ECMWF Open IFS ensemble analysis and prepare
the WeatherMesh-5c background:

```bash
./scripts/wm5c-uv.sh run wm5c-background ingest --member 1
```

Use `--cycle YYYYMMDDHH` for a fixed 00, 06, 12, or 18 UTC cycle. `fetch` and `process`
can run the two stages separately. Raw files go to `data/raw`, and prepared backgrounds
go to `data/backgrounds` by default. Open perturbed members 1 through 50 are supported.

The adapter makes these explicit conversions:

- converts open geopotential height `gh` to geopotential `z` with standard gravity;
- interpolates 30 and 70 hPa in log pressure from the adjacent open levels;
- interpolates specific humidity in log value so that it stays positive;
- applies the checkpoint's level-specific normalization;
- expands the 16 model levels to the encoder's 25 levels with the legacy calculation;
- searches back up to seven days for the latest available earlier NOAA OISST field,
  with Open IFS skin temperature only for missing ocean pixels;
- uses the Open IFS land-sea mask to keep the normalized SST channel zero over land.

The prepared file records its IFS cycle, ensemble member, and OISST date. AIDA uses the
background cycle as its initial time. The full runner assimilates observations at
successive three-hour steps as described above. The single-update smoke test below
uses an AIDA valid time three hours after the background cycle.

Run a background-only AIDA and forecast smoke test:

```bash
./scripts/wm5c-uv.sh run --no-sync python -m weathermesh5c.smoke \
  --bundle-root /path/to/runtime-bundle \
  --device cuda:0 \
  --background /path/to/prepared-background.npz \
  --decode
```

Add `--observations-root data/processed` to use processed public observations. The
smoke test infers the AIDA valid time from the background unless `--valid-time` replaces
it.

## Observation ingestion

The NOAA observation files use NCEP BUFR with embedded tables. Install the official
NOAA decoder in the active virtual environment:

```bash
./scripts/wm5c-uv.sh sync --extra dev --extra ingest-build
./scripts/wm5c-uv.sh run --no-sync wm5c-install-ncepbufr
```

The installer pins NCEPLIBS-bufr `v12.3.0` at commit
`328f2294082ad1fac27525859b8422882295671a`. It needs CMake, a C compiler, and a
Fortran compiler. It refuses to install into a system Python prefix. Package consumers
can run `wm5c-install-ncepbufr` directly in their WeatherMesh virtual environment.

The microwave, IASI, and satellite-wind readers reuse numeric buffers through the
pinned decoder's native entry points. This avoids repeated masked-array allocation
without changing decoding or missing-value rules. Other bindings use the public reader.

Download and process one current GDAS cycle:

```bash
./scripts/wm5c-uv.sh run wm5c-obs ingest \
  1bamua 1bmhs atms gpsro mtiasi satwnd adpupa adpsfc
```

Use `--cycle YYYYMMDDHH` to select a cycle. Use `fetch` and `process` instead of
`ingest` when download and processing must run as separate jobs. Raw data goes to
`data/raw` and processed hourly files go to `data/processed` by default.

`wm5c-run` prepares the background and observations in parallel. For NOAA sources,
it reuses processed data when the raw file and all hourly output files are unchanged.
It stores processing receipts in `data/processed/.ingested`. Remove a receipt to force
that cycle to be processed again. Raw NOAA files are stored by date so that cycles
from different days cannot share a cache entry. Existing month-only raw caches are
not reused, since their filenames do not identify the day.

The pinned NOAA library decodes every listed BUFR source, including ADPSFC surface
reports. Surface ingestion runs inside this package and needs no separate executable or
configuration file.

To add WindBorne observations, install the official API client and set the API key:

```bash
./scripts/wm5c-uv.sh sync --extra dev --extra ingest-build --extra windborne
export WB_API_KEY=wb_your_combined_api_key
./scripts/wm5c-uv.sh run --no-sync wm5c-obs ingest all \
  --cycle YYYYMMDDHH \
  --windborne
```

The WindBorne source uses official super-observations, the same source type used by the
operational 5C pipeline. It fetches the complete three-hour AIDA window in bounded
10-minute requests, follows API pagination, and removes duplicate observation IDs. The
converter keeps the API's relative-humidity and specific-humidity units because those
units are part of the trained 5C input contract. No credential is stored in the package.

An exported API CSV remains supported through `wm5c-obs process windborne FILE.csv`.

## Smoke test

The basic smoke test uses a normalized synthetic background and empty observation
batches. It runs the background encoder, AIDA, and one three-hour forecast processor
step. It does not make a meteorologically useful forecast.

```bash
./scripts/wm5c-uv.sh run --no-sync python -m weathermesh5c.smoke \
  --bundle-root /path/to/runtime-bundle \
  --device cuda:0
```

After observation ingestion, test the complete raw-data-to-AIDA path. The valid time
must be an exact UTC hour:

```bash
./scripts/wm5c-uv.sh run --no-sync python -m weathermesh5c.smoke \
  --bundle-root /path/to/runtime-bundle \
  --device cuda:0 \
  --observations-root data/processed \
  --valid-time YYYYMMDDHH \
  --max-observations 20000 \
  --decode
```

Add `--background PATH` to replace the synthetic background with a prepared Open IFS
background. Omit `--max-observations` to retain every eligible observation.

Run these checks before each push:

```bash
./scripts/wm5c-uv.sh run --no-sync ruff check
./scripts/wm5c-uv.sh run --no-sync pytest
./scripts/wm5c-uv.sh lock --check
```

## License

The inference runtime is licensed under [Apache-2.0](LICENSE). See
[NOTICE.md](NOTICE.md) for third-party notices. The checkpoint and bundled assets
are covered separately by the terms on the
[Hugging Face model card](https://huggingface.co/windborne/weathermesh-5c).

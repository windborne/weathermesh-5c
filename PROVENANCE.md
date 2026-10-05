# Provenance

## Model

- Product: WeatherMesh-5c with AIDA
- Product version: `2b81`
- Source repository: WindBorne `deep`
- Source commit: `5f41011701988aa84f05279041a6eb2a0b3a3783`
- Source builder: `package/package_clean.py:get_armstrong_coup`

## Release checkpoint

- File: `weather-mesh-5c-aida.pt`
- Size: `2841543169` bytes
- SHA-256: `20d6a06cf75f6fa42c8796bc39fa7e426ea08be390a895b6236a6f03aabe73d2`
- Format: plain PyTorch state dictionary
- Entries: `984`
- Tensor elements: `762997823`

The state dictionary contains the analysis encoder, observation encoders, AIDA
transformer, forecast processor, ensemble forker, predecoder, and global decoder. It is
the exact state required by the release model and loads with strict key matching.

## Runtime bundle

The checkpoint and production runtime assets are delivered as one external bundle with
a versioned JSON manifest. The manifest binds every file to its role, byte count, and
SHA-256. It contains no machine-specific path. Code receives one bundle root through
the Python API or command line.

The bundle builder converts the trusted legacy HEALPix mesh pickles to numeric `.npz`
files. The runtime uses NumPy with pickle disabled. No inference path imports or opens
pickle data. Solar elevation and direct radiation come from the original deterministic
production formula. The runtime generates only the fields needed for the requested UTC
timestamps and caches them in memory.

The schema-version 3 reference bundle has 10 files and 3,535,775,581 bytes.

## Inference core

`weathermesh5c/_core` contains only the runtime needed to construct and execute this
model. The package uses `checkpoint_type="none"` and has no training-time activation
checkpoint implementation. It preserves the release checkpoint key contract and the
source model's inference calculations.

## Operational ensemble contract

The ensemble schedule and products follow the operational producer and aggregator
at the source commit. The runner preserves the 24 input analyses, learned eight-way
fork at hour 72, surface and pressure schedules, pressure supermembers, precipitation
quantile mean, derived wind speeds, and overall sample standard deviation. It uses
a streamed single-GPU calculation and lossless compressed Zarr output by default.
Compressed and uncompressed NetCDF are also supported. It retains the operational
bounded pinned-memory transfer, compact float16 aggregation representation, mean-only
member stores, and independent output workers.

## Observation pipeline

The observation schemas, normalization, NOAA NOMADS naming, and BUFR processing logic
come from the source commit. The package keeps the model-facing arrays and calculations.
All NOAA sources, including ADPSFC surface reports, use NOAA NCEPLIBS-bufr `v12.3.0`
at commit `328f2294082ad1fac27525859b8422882295671a`. The native surface mapping preserves
the output contract of UCAR's MIT-licensed
[`rda-bufr-decode-ADPsfc`](https://github.com/NCAR/rda-bufr-decode-ADPsfc) at commit
`2e72b720f67104423a928db0a70d604144b4fda3`. No UCAR executable or source is bundled.

NCEPLIBS-bufr is a separately built dependency, not bundled runtime code. Its pinned
source is licensed under LGPL-3.0; see the
[NOAA license notice](https://github.com/NOAA-EMC/NCEPLIBS-bufr/blob/328f2294082ad1fac27525859b8422882295671a/LICENSE.md).

The source IASI PCA and scaler have SHA-256 values
`d953768c59883c2d8f4c864a1096d1fff273ef2533b43fb02dd8aaa86d17d544` and
`c80418c766e75385880558c43838f79c90d55ee2de6e4966942d963540b808e5`. They are
stored as the equivalent projection matrix and bias, without executable pickle data,
at SHA-256 `01a85dad1c372c5dbe24c0de163b802384b95d2f2c503114d289d4df4794bd6f`.

## Open background pipeline

The background tensor contract and 16-to-25 pressure-level expansion come from
the source commit. The adapter downloads the public 0.25-degree IFS ensemble feed from
ECMWF and daily 0.25-degree OISST from NOAA. It does not read, package, or redistribute WindBorne's
paid ECMWF delivery. The 30 and 70 hPa interpolation and OISST substitution are new
release-adapter calculations validated against the internal paid input.

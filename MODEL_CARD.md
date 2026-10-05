---
license: cc-by-4.0
library_name: earth2studio
tags:
  - weather
  - forecasting
  - data-assimilation
  - earth2studio
---

# WeatherMesh-5c with AIDA

This model package contains the WeatherMesh-5c inference checkpoint and the static
runtime assets required by the Earth2Studio integration.

The package contains:

- the inference-only WeatherMesh-5c and AIDA checkpoint;
- global mesh and static surface assets;
- a manifest with the size and SHA-256 of every runtime file.

The source package generates the deterministic solar and radiation inputs for each
forecast timestamp.

## Model and outputs

WeatherMesh-5c is a PyTorch global forecast model. AIDA creates its initial state
from an IFS ensemble background and observations. The forecast model advances a
native HEALPix latent state in three-hour steps. Its decoder produces physical
fields on a 720 by 1440 global latitude-longitude grid.

The coupled workflow supports forecasts through hour 360. The operational schedule
uses 24 initial members and an eight-way learned fork at hour 72. One initial
member and a smaller branch count can be selected for testing or limited resources.

Compressed Zarr v2 ZIP stores are the default output. Compressed and uncompressed
NetCDF remain available. Output products include surface and pressure-level fields,
ensemble means, and sample standard deviations.

## Inputs

The public-data workflow uses ECMWF Open IFS, NOAA OISST, and public NOAA
observations. WindBorne observations are optional and require separate API access.
They are not included in the model package. Forecast input caches are not part of
this model package either.

## Package and use

The schema-version 3 bundle contains ten checkpoint and asset files. Their total
size is 3,535,775,581 bytes, about 3.54 GB. The manifest records their relative
paths, roles, sizes, and SHA-256 checksums.

The checkpoint contains 984 state entries and 762,997,823 tensor elements. The
loader uses strict key matching and tensor-only loading. Runtime mesh assets are
numeric NumPy files; inference does not load executable pickle data.

Use the [runtime](https://github.com/windborne/weathermesh-5c) or its Earth2Studio
integration. See the runtime README for installation, model loading, and forecast
commands. A CUDA GPU is required for the complete workflow. Observation volume,
member count, and selected products affect memory, time, and disk requirements.

## Validation and limits

See the [validation record](https://github.com/windborne/weathermesh-5c/blob/main/VALIDATION.md)
for fixed-input comparisons against the native implementation. These checks
establish numerical agreement for the stated cases.
They do not establish forecast skill, live-feed availability, or bit-for-bit
agreement across different GPU types.

This package does not include training code, training data, or an optimizer state.

## License

The WeatherMesh-5c with AIDA checkpoint and WindBorne-authored assets in this
bundle are available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Third-party material, if identified with separate terms, retains those terms.
Attribute the release to **WindBorne Systems, Inc.** and link to this model card.
The [inference runtime](https://github.com/windborne/weathermesh-5c) is licensed
separately under Apache-2.0. This release does not grant access to WindBorne's
private observation API or rights to redistribute input datasets.

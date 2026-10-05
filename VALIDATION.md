# Numerical validation

The runtime was checked against the canonical source and checkpoint described in
[PROVENANCE.md](PROVENANCE.md). The bundle manifest, file checksums, and strict
checkpoint loading were also checked.

## Fixed-input comparisons

Comparisons used an RTX 4090, PyTorch 2.11.0, and CUDA 12.8. Prepared observations,
backgrounds, model weights, and random seeds were held fixed.

The Earth2Studio workflow matched the reference exactly for all 585 checked fields,
with no outlier fields. Source checkouts and built wheels passed. The case used two
uncapped AIDA updates, one input member, and two learned branches at hour 12. Surface
outputs were checked at hours 0, 12, and 15; pressure outputs at hours 0 and 12.

A separate native-source comparison matched exactly at the AIDA, forecast, decoder,
and learned-fork boundaries. It checked 13 analysis and rollout arrays and 28 fork
arrays: 227,105,280 values in total. The checks covered temperature at hours 0, 12,
and 24, plus forks at hours 12 and 72 with two branches and offsets of 0, 3, and
12 hours. Fork inputs, noise, and state hashes also matched.

## Full global forecast

A cached-input forecast completed through hour 360 with three uncapped public
AIDA updates, one starting member, eight learned branches at hour 72, and seed
zero. It wrote surface products every three hours and pressure products on the
native six-hour UTC cadence through hour 120.

On an RTX 4090 and AMD EPYC 7642, with 20 Torch CPU threads and four output
workers, the optimized runtime took 328.4 seconds, versus 976.4 seconds before
the CPU statistics changes: 2.97 times faster. This includes cached input and
model loading, AIDA, rollout, decoding, statistics, and all output writes.
Input checks and the final output audit are outside this time.

All 262 lossless Zarr stores matched the earlier outputs exactly, including
13,045,017,600 float32 forecast values, coordinates, and field metadata.
The built runtime also matched the 585-field reference and saved native-source
arrays described above. CPU tests checked the real Earth2Studio workflow with
the optimized runtime. These measurements are not DGX Spark results.

## Scope

These comparisons use prepared inputs. They are not an independent comparison of
the full raw-data ingestion path or live feeds. They do not cover a full 24-member,
15-day forecast or guarantee bit-for-bit agreement across GPU types.

Runtime and WeatherMesh integration tests cover observation handling, AIDA state
layout, ensemble scheduling, product aggregation, package loading, and the
Earth2Studio interface. They do not replace the complete Earth2Studio all-model
and data-download test suite.

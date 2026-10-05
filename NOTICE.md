# WeatherMesh-5c inference runtime

Copyright 2026 WindBorne Systems, Inc.

The WindBorne inference runtime is licensed under Apache-2.0; see [LICENSE](LICENSE).
This license does not apply to the separately distributed model checkpoint and bundle,
or to data obtained through external services.

## Third-party notices

- The optional NOAA BUFR installer fetches [NCEPLIBS-bufr v12.3.0](https://github.com/NOAA-EMC/NCEPLIBS-bufr/tree/328f2294082ad1fac27525859b8422882295671a).
  The separately built decoder is distributed by NOAA under the [GNU Lesser General
  Public License v3.0](https://github.com/NOAA-EMC/NCEPLIBS-bufr/blob/328f2294082ad1fac27525859b8422882295671a/LICENSE.md).
  Neither its source nor its binaries are included in this package.
- The surface observation mapping follows the output of UCAR's
  [MIT-licensed ADPSFC decoder](https://github.com/NCAR/rda-bufr-decode-ADPsfc/tree/2e72b720f67104423a928db0a70d604144b4fda3).
  No UCAR source or executable is bundled.
- Python dependencies are installed separately and retain their own licenses.

See [PROVENANCE.md](PROVENANCE.md) for source and asset provenance.

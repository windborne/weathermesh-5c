"""Deep-style packed forecast arrays in a single, lossless Zarr v2 ZIP store."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import zarr
from numcodecs import Blosc

from .output import (
    DERIVED_SURFACE_FIELDS,
    PRESSURE_FIELDS,
    SURFACE_FIELDS,
    OutputGrid,
    PhysicalProducts,
    _field_attributes,
)


def _create_coordinate(
    root: zarr.Group,
    name: str,
    data: np.ndarray,
    dimensions: tuple[str, ...],
    attributes: dict[str, object],
) -> zarr.Array:
    metadata = {"_ARRAY_DIMENSIONS": list(dimensions), **attributes}
    options = {
        "chunks": data.shape,
        "compressor": None,
        "fill_value": None,
    }
    if hasattr(root, "create_array"):
        return root.create_array(name, data=data, attributes=metadata, **options)
    array = root.create_dataset(name, data=data, **options)
    array.attrs.update(metadata)
    return array


def write_zarr_products(
    path: str,
    products: tuple[PhysicalProducts, ...],
    labels: tuple[str, ...],
    mesh: OutputGrid,
    valid_time: datetime,
    attributes: dict[str, object],
    *,
    compression: bool,
) -> None:
    """Keep Deep's packed-variable chunks and codec, without its value quantization."""

    modern = int(zarr.__version__.split(".")[0]) >= 3
    # Packed arrays contain fields with different units; do not claim CF data-variable conformance.
    attributes = {name: value for name, value in attributes.items() if name != "Conventions"}
    options = {"zarr_format": 2, "attributes": attributes} if modern else {"zarr_version": 2}
    codec = Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE) if compression else None
    multiple = len(products) > 1
    prefix = ("ensemble_statistic",) if multiple else ()
    prefix_shape = (len(products),) if multiple else ()
    nlat, nlon = len(mesh.lats), len(mesh.lons)
    with zarr.storage.ZipStore(path, mode="w") as store:
        root = zarr.group(store=store, **options)
        if not modern:
            root.attrs.update(attributes)
        for name, data, metadata in (
            (
                "time",
                np.asarray([int(valid_time.timestamp())], dtype=np.int64),
                {
                    "standard_name": "time",
                    "units": "seconds since 1970-01-01 00:00:00 UTC",
                    "calendar": "standard",
                },
            ),
            (
                "latitude",
                np.asarray(mesh.lats, dtype=np.float32),
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            (
                "longitude",
                np.asarray(mesh.lons, dtype=np.float32),
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
        ):
            _create_coordinate(root, name, data, (name,), metadata)
        if multiple:
            _create_coordinate(root, "ensemble_statistic", np.asarray(labels), prefix, {})
        for kind, fields in (
            ("surface", (*SURFACE_FIELDS, *DERIVED_SURFACE_FIELDS)),
            ("pressure", PRESSURE_FIELDS),
        ):
            values = [getattr(product, kind) for product in products]
            if all(value is None for value in values):
                continue
            if any(value is None for value in values):
                raise ValueError(
                    f"{kind.capitalize()} product availability differs between statistics"
                )
            metadata = [
                {
                    "name": field.name,
                    **_field_attributes(field),
                    **({"pressure_hpa": int(level)} if level is not None else {}),
                }
                for field in fields
                for level in (mesh.levels if kind == "pressure" else (None,))
            ]
            names = [
                f"{item['name']}_{item['pressure_hpa']}" if kind == "pressure" else item["name"]
                for item in metadata
            ]
            variable_dimension = f"{kind}_variable"
            _create_coordinate(
                root, variable_dimension, np.asarray(names), (variable_dimension,), {}
            )
            shape = (*prefix_shape, 1, len(names), nlat, nlon)
            dimensions = (*prefix, "time", variable_dimension, "latitude", "longitude")
            array_metadata = {"_ARRAY_DIMENSIONS": list(dimensions), "variable_metadata": metadata}
            create_options = {"attributes": array_metadata} if modern else {}
            array = root.create(
                kind,
                shape=shape,
                dtype="f4",
                chunks=(1,) * (len(prefix) + 1) + (len(names), 45, 45),
                compressor=codec,
                fill_value=None,
                **create_options,
            )
            if not modern:
                array.attrs.update(array_metadata)
            for index, value in enumerate(values):
                expected_shape = (
                    (len(fields), len(mesh.levels), nlat, nlon)
                    if kind == "pressure"
                    else (len(fields), nlat, nlon)
                )
                if value.shape != expected_shape:
                    raise ValueError(
                        f"{kind.capitalize()} product is {value.shape}, expected {expected_shape}"
                    )
                selection = (index, 0) if multiple else (0,)
                array[selection] = np.asarray(value, dtype=np.float32).reshape(
                    len(names), nlat, nlon
                )

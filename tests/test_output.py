from __future__ import annotations

import json
from datetime import datetime, timezone
from zipfile import ZipFile

import netCDF4
import numpy as np
import pytest
import xarray as xr
import zarr

from weathermesh5c.ensemble import DiskStatistics
from weathermesh5c.output import (
    DERIVED_SURFACE_FIELDS,
    PRESSURE_FIELDS,
    SURFACE_FIELDS,
    OutputGrid,
    PhysicalProducts,
    ProductTransform,
    write_product_file,
)
from weathermesh5c.run import (
    ForecastRequest,
    _expected_outputs,
    _output_executor,
    _request_from_args,
    _write_jobs,
    build_parser,
)
from weathermesh5c.writer import ProductFileJob


@pytest.fixture
def forecast():
    grid = OutputGrid(
        (10, 500, 1000),
        np.arange(46, dtype=np.float32),
        np.arange(47, dtype=np.float32),
    )
    rng = np.random.default_rng(0)
    products = []
    for _ in range(2):
        surface = rng.standard_normal((14, 46, 47), dtype=np.float32)
        pressure = rng.standard_normal((5, 3, 46, 47), dtype=np.float32)
        surface[0, 0, :2] = [0, np.nan]
        pressure[0, 0, 0, :2] = [0, np.nan]
        products.append(PhysicalProducts(surface, pressure))
    return grid, tuple(products)


def _write(path, grid, products, *, compression=True, **metadata):
    return write_product_file(
        path,
        products,
        ("mean", "standard_deviation") if len(products) > 1 else ("member_mean",),
        grid,
        background_time=datetime(2026, 9, 21, 0, tzinfo=timezone.utc),
        aida_valid_time=datetime(2026, 9, 21, 9, tzinfo=timezone.utc),
        lead_hours=12,
        trajectory_count=2,
        compression=compression,
        **metadata,
    )


@pytest.mark.parametrize("entry_point", ["python", "cli"])
@pytest.mark.parametrize(
    "output_format, compression", [(None, True), ("netcdf", True), ("netcdf", False)]
)
def test_requested_output_format_writes_readable_forecast(
    tmp_path, forecast, entry_point, output_format, compression
):
    if entry_point == "cli":
        arguments = ["--bundle-root", str(tmp_path)]
        if output_format:
            arguments.extend(("--output-format", output_format))
        if not compression:
            arguments.append("--no-output-compression")
        request = _request_from_args(build_parser().parse_args(arguments))
    else:
        options = {"output_format": output_format} if output_format else {}
        request = ForecastRequest(output_compression=compression, **options)
    grid, products = forecast
    path = _expected_outputs(tmp_path, (12,), request.output_format)[0]
    result = _write(path, grid, products, compression=request.output_compression)
    if output_format == "netcdf":
        with netCDF4.Dataset(result.path) as dataset:
            field = dataset["air_temperature_2m"]
            assert field.filters()["zlib"] == compression
            temperature = field[:, 0]
    else:
        with ZipFile(result.path) as archive:
            codec = json.loads(archive.read("surface/.zarray"))["compressor"]
            assert codec["id"] == "blosc" and codec["cname"] == "zstd"
        with zarr.storage.ZipStore(str(result.path), mode="r") as store:
            with xr.open_zarr(store, consolidated=False) as dataset:
                temperature = dataset.surface.sel(surface_variable="air_temperature_2m").values[
                    :, 0
                ]
    np.testing.assert_array_equal(
        temperature, np.stack([product.surface[2] for product in products])
    )


@pytest.mark.parametrize("compression", [True, False])
@pytest.mark.parametrize("multiple", [True, False])
@pytest.mark.parametrize("kind", ["all", "surface", "pressure"])
def test_zarr_and_netcdf_preserve_products_coordinates_and_metadata(
    tmp_path, forecast, compression, multiple, kind
):
    grid, products = forecast
    products = tuple(
        PhysicalProducts(
            product.surface if kind != "pressure" else None,
            product.pressure if kind != "surface" else None,
        )
        for product in (products if multiple else products[:1])
    )
    metadata = {} if multiple else {"input_member": 7, "pressure_supermember": 2}
    netcdf = _write(tmp_path / "forecast.nc", grid, products, compression=compression, **metadata)
    packed = _write(tmp_path / "forecast.zarr", grid, products, compression=compression, **metadata)
    assert packed.valid_time == netcdf.valid_time
    assert packed.size_bytes == packed.path.stat().st_size

    with (
        netCDF4.Dataset(netcdf.path) as reference,
        zarr.storage.ZipStore(str(packed.path), mode="r") as store,
    ):
        reference.set_auto_mask(False)
        actual = zarr.open_group(store=store, mode="r")
        assert dict(actual.attrs) == {
            name: reference.getncattr(name) for name in reference.ncattrs() if name != "Conventions"
        }
        for coordinate in ("time", "latitude", "longitude"):
            np.testing.assert_array_equal(actual[coordinate][:], reference[coordinate][:])
        if multiple:
            np.testing.assert_array_equal(
                actual["ensemble_statistic"][:], reference["ensemble_statistic"][:]
            )
        for product_kind, fields in (
            ("surface", (*SURFACE_FIELDS, *DERIVED_SURFACE_FIELDS)),
            ("pressure", PRESSURE_FIELDS),
        ):
            if getattr(products[0], product_kind) is None:
                assert product_kind not in actual
                continue
            array = actual[product_kind]
            for channel, field_metadata in enumerate(array.attrs["variable_metadata"]):
                field = reference[field_metadata["name"]]
                for key in field.ncattrs():
                    if key != "_FillValue":
                        assert field_metadata[key] == field.getncattr(key)
                expected = field[:]
                if product_kind == "pressure":
                    level = list(grid.levels).index(field_metadata["pressure_hpa"])
                    expected = expected[..., level, :, :]
                np.testing.assert_array_equal(array[..., channel, :, :], expected)
            assert len(array.attrs["variable_metadata"]) == len(fields) * (
                len(grid.levels) if product_kind == "pressure" else 1
            )


def test_zarr_zip_is_xarray_readable_without_masking_zero_coordinates(tmp_path, forecast):
    grid, products = forecast
    result = _write(tmp_path / "forecast.zarr", grid, products)
    with zarr.storage.ZipStore(str(result.path), mode="r") as store:
        with xr.open_zarr(store, consolidated=False) as dataset:
            np.testing.assert_array_equal(dataset.latitude.values, grid.lats)
            np.testing.assert_array_equal(dataset.longitude.values, grid.lons)
            np.testing.assert_array_equal(
                dataset.surface.sel(surface_variable="air_temperature_2m").values[:, 0],
                np.stack([product.surface[2] for product in products]),
            )
            np.testing.assert_array_equal(
                dataset.pressure.sel(pressure_variable="air_temperature_500").values[:, 0],
                np.stack([product.pressure[1, 1] for product in products]),
            )


@pytest.mark.parametrize("suffix", ["nc", "zarr"])
def test_failed_write_preserves_existing_forecast_and_removes_partial_file(
    tmp_path, forecast, suffix
):
    grid, products = forecast
    path = tmp_path / f"forecast.{suffix}"
    _write(path, grid, products)
    previous = path.read_bytes()
    broken = (products[0], PhysicalProducts(pressure=products[1].pressure))
    with pytest.raises(ValueError, match="availability differs"):
        _write(path, grid, broken)
    assert path.read_bytes() == previous
    assert sorted(item.name for item in tmp_path.iterdir()) == [path.name]


@pytest.mark.parametrize("suffix", ["nc", "zarr"])
def test_statistics_jobs_honor_compression_without_changing_values(tmp_path, forecast, suffix):
    grid, products = forecast
    values = PhysicalProducts(np.ones_like(products[0].surface), np.ones_like(products[0].pressure))
    statistics = DiskStatistics(tmp_path / "statistics", mean_only=True)
    statistics.add(12, values)
    statistics.flush()
    jobs = [
        ProductFileJob(
            path=tmp_path / f"compressed-{compression}.{suffix}",
            snapshots=(statistics.reference(12, "mean"),),
            labels=("member_mean",),
            product_kind="all",
            grid=grid,
            transform=ProductTransform(np.zeros(27, np.float32), np.ones(27, np.float32)),
            background_time=datetime(2026, 9, 21, 0, tzinfo=timezone.utc),
            aida_valid_time=datetime(2026, 9, 21, 9, tzinfo=timezone.utc),
            lead_hours=12,
            trajectory_count=1,
            compression=compression,
        )
        for compression in (True, False)
    ]
    with _output_executor(2) as executor:
        results = _write_jobs(executor, jobs)
    paths = []
    for result in results:
        paths.append(result.path)
        if suffix == "nc":
            with netCDF4.Dataset(result.path) as dataset:
                np.testing.assert_array_equal(dataset["air_temperature_2m"][:], values.surface[2:3])
        else:
            with zarr.storage.ZipStore(str(result.path), mode="r") as store:
                dataset = zarr.open_group(store=store, mode="r")
                np.testing.assert_array_equal(dataset["surface"][0], values.surface)
    assert paths[0].stat().st_size < paths[1].stat().st_size / 2
    statistics.close()

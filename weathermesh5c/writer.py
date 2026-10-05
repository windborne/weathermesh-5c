from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path

import numpy as np

from .ensemble import StatisticsSnapshot
from .output import (
    ForecastFile,
    OutputGrid,
    PhysicalProducts,
    ProductTransform,
    physicalize_statistics,
    write_product_file,
)


@dataclass(frozen=True)
class ProductFileJob:
    path: Path
    snapshots: tuple[StatisticsSnapshot, ...]
    labels: tuple[str, ...]
    product_kind: str
    grid: OutputGrid
    transform: ProductTransform
    background_time: datetime
    aida_valid_time: datetime
    lead_hours: int
    trajectory_count: int
    input_member: int | None = None
    pressure_supermember: int | None = None
    compression: bool = True


def _load_snapshot(snapshot: StatisticsSnapshot) -> PhysicalProducts:
    values = {}
    for field in fields(PhysicalProducts):
        array = snapshot.arrays.get(field.name)
        if array is None:
            values[field.name] = None
            continue
        mean = np.load(array.mean, mmap_mode="r")
        if array.axes:
            mean = mean.transpose(array.axes)
        if snapshot.statistic == "mean":
            values[field.name] = mean
            continue
        if array.squared_difference is None:
            raise ValueError("Standard deviation state is missing")
        if array.count < 2:
            values[field.name] = np.zeros(mean.shape, dtype=np.float32)
            continue
        squared_difference = np.load(array.squared_difference, mmap_mode="r")
        if array.axes:
            squared_difference = squared_difference.transpose(array.axes)
        variance = squared_difference / (array.count - 1)
        np.sqrt(variance, out=variance)
        values[field.name] = variance
    return PhysicalProducts(**values)


def _select_products(products: PhysicalProducts, kind: str) -> PhysicalProducts:
    if kind == "all":
        return products
    if kind == "surface":
        return PhysicalProducts(surface=products.surface)
    if kind == "pressure":
        return PhysicalProducts(pressure=products.pressure)
    raise ValueError(f"Unknown product kind: {kind}")


def write_statistics_file(job: ProductFileJob) -> ForecastFile:
    products = tuple(
        physicalize_statistics(
            _select_products(_load_snapshot(snapshot), job.product_kind),
            job.transform,
            snapshot.statistic,
        )
        for snapshot in job.snapshots
    )
    return write_product_file(
        job.path,
        products,
        job.labels,
        job.grid,
        background_time=job.background_time,
        aida_valid_time=job.aida_valid_time,
        lead_hours=job.lead_hours,
        input_member=job.input_member,
        pressure_supermember=job.pressure_supermember,
        trajectory_count=job.trajectory_count,
        compression=job.compression,
    )

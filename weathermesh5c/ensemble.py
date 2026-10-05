from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

from .output import (
    DecodedProducts,
    PhysicalProducts,
    select_global_products,
)

OPERATIONAL_MEMBERS = tuple(range(1, 25))
OPERATIONAL_SURFACE_LEADS = tuple(range(0, 15 * 24 + 1, 3))
OPERATIONAL_PRESSURE_LEADS = tuple(range(0, 5 * 24 + 1, 6))
OPERATIONAL_FORK_HOUR = 72
OPERATIONAL_FORK_COUNT = 8


@dataclass(frozen=True)
class EnsemblePlan:
    members: tuple[int, ...] = OPERATIONAL_MEMBERS
    surface_leads: tuple[int, ...] = OPERATIONAL_SURFACE_LEADS
    pressure_leads: tuple[int, ...] = OPERATIONAL_PRESSURE_LEADS
    fork_hour: int = OPERATIONAL_FORK_HOUR
    fork_count: int = OPERATIONAL_FORK_COUNT
    seed: int = 0
    pressure_group_size: int = 6

    def __post_init__(self) -> None:
        if not self.members or len(set(self.members)) != len(self.members):
            raise ValueError("Ensemble members must be non-empty and unique")
        if any(not 1 <= member <= 50 for member in self.members):
            raise ValueError("Open IFS perturbed members are numbered 1 through 50")
        if not self.surface_leads or tuple(sorted(set(self.surface_leads))) != self.surface_leads:
            raise ValueError("Surface leads must be sorted and unique")
        if self.surface_leads[0] < 0 or any(lead % 3 for lead in self.surface_leads):
            raise ValueError("Surface leads must be non-negative multiples of three")
        if tuple(sorted(set(self.pressure_leads))) != self.pressure_leads:
            raise ValueError("Pressure leads must be sorted and unique")
        if (
            any(lead not in self.surface_leads for lead in self.pressure_leads)
            or len({lead % 6 for lead in self.pressure_leads}) > 1
        ):
            raise ValueError("Pressure leads must be six-hour surface leads")
        if self.fork_hour < 12 or self.fork_hour % 12:
            raise ValueError("The learned fork hour must be a positive multiple of 12")
        if self.fork_count < 1:
            raise ValueError("The learned fork count must be positive")
        if self.pressure_group_size < 1:
            raise ValueError("The pressure group size must be positive")

    def trajectory_count(self, lead: int) -> int:
        branches = self.fork_count if lead >= self.fork_hour else 1
        return len(self.members) * branches


def decode_products_for_transfer(
    model,
    state: torch.Tensor,
    *,
    include_pressure: bool,
) -> DecodedProducts:
    """Decode only the compact tensors sent by the operational producer."""

    global_latent = model.predecoder(state, sfc_only=not include_pressure)
    decoded = model.decoders[0](global_latent)
    surface, pressure = select_global_products(
        decoded,
        model.decoders[0].mesh,
        surface_only=not include_pressure,
    )
    del decoded, global_latent

    return DecodedProducts(surface=surface, pressure=pressure)


def _set_fork_seed(seed: int, member: int, branch: int) -> None:
    value = seed + member * 1_000_003 + branch
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def rollout_member(
    model,
    state: torch.Tensor,
    plan: EnsemblePlan,
    member: int,
    emit: Callable[[int, int | None, torch.Tensor], None],
) -> None:
    """Stream one input member with the operational learned-fork schedule."""

    if member not in plan.members:
        raise ValueError(f"Member {member} is not in the ensemble plan")
    processor = model.processors["3"]
    last_lead = plan.surface_leads[-1]
    pre_fork_leads = tuple(lead for lead in plan.surface_leads if lead < plan.fork_hour)
    current_lead = 0
    current_state = state
    for lead in pre_fork_leads:
        while current_lead < lead:
            current_state = processor(current_state)
            current_lead += 3
        emit(lead, None, current_state)

    if last_lead < plan.fork_hour:
        return
    while current_lead < plan.fork_hour:
        current_state = processor(current_state)
        current_lead += 3
    fork_source = current_state
    branch_leads = tuple(lead for lead in plan.surface_leads if lead >= plan.fork_hour)
    for branch in range(plan.fork_count):
        _set_fork_seed(plan.seed, member, branch)
        branch_state = model.forker(fork_source.clone(), 1, plan.fork_hour)
        branch_lead = plan.fork_hour
        for lead in branch_leads:
            while branch_lead < lead:
                branch_state = processor(branch_state)
                branch_lead += 3
            emit(lead, branch, branch_state)


@dataclass
class _RunningArray:
    mean: np.memmap
    squared_difference: np.memmap | None
    count: int = 0
    storage_axes: tuple[int, ...] = ()

    def add(self, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=np.float32)
        if value.shape != self.mean.shape:
            raise ValueError(f"Product shape {value.shape} changed from {self.mean.shape}")
        self.count += 1
        if self.count == 1:
            # Adding zero preserves the existing first update's signed-zero behavior.
            np.add(value, np.float32(0), out=self.mean)
            return
        axes = self.storage_axes or tuple(range(value.ndim))
        value = np.require(value.transpose(axes), requirements=["C", "W"]).reshape(-1)
        mean = np.asarray(self.mean.transpose(axes)).reshape(-1)
        m2 = (
            None
            if self.squared_difference is None
            else np.asarray(self.squared_difference.transpose(axes)).reshape(-1)
        )
        assert np.shares_memory(mean, self.mean), (
            "Statistics storage must remain a file-backed view"
        )
        value, mean = torch.from_numpy(value), torch.from_numpy(mean)
        m2 = None if m2 is None else torch.from_numpy(m2)
        # A one-element divisor preserves NumPy's division rounding.
        divisor = torch.tensor([self.count], dtype=torch.float32, device="cpu")
        block = 4_194_304
        difference = torch.empty(min(block, value.numel()), dtype=torch.float32, device="cpu")
        scratch = torch.empty_like(difference)
        for start in range(0, value.numel(), block):
            stop = min(start + block, value.numel())
            x, average = value[start:stop], mean[start:stop]
            delta, temporary = difference[: stop - start], scratch[: stop - start]
            torch.sub(x, average, out=delta)
            torch.div(delta, divisor, out=temporary)
            torch.add(average, temporary, out=average)
            if m2 is not None:
                torch.sub(x, average, out=temporary)
                torch.mul(delta, temporary, out=temporary)
                # Variance state is first needed on the second sample.
                previous = 0 if self.count == 2 else m2[start:stop]
                torch.add(temporary, previous, out=m2[start:stop])

    def standard_deviation(self) -> np.ndarray:
        if self.squared_difference is None:
            raise ValueError("This accumulator tracks the mean only")
        if self.count < 2:
            return np.zeros(self.mean.shape, dtype=np.float32)
        variance = self.squared_difference / (self.count - 1)
        np.sqrt(variance, out=variance)
        return variance


@dataclass(frozen=True)
class StatisticsArray:
    mean: Path
    squared_difference: Path | None
    count: int
    axes: tuple[int, ...] = ()


@dataclass(frozen=True)
class StatisticsSnapshot:
    arrays: dict[str, StatisticsArray]
    statistic: str


class DiskStatistics:
    """Welford statistics backed by local files instead of forecast-sized RAM."""

    def __init__(self, root: str | Path, *, mean_only: bool = False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.mean_only = mean_only
        self._arrays: dict[tuple[int, str], _RunningArray] = {}

    def _array(self, lead: int, name: str, value: np.ndarray) -> _RunningArray:
        key = (lead, name)
        if key not in self._arrays:
            directory = self.root / f"f{lead:03d}"
            directory.mkdir(parents=True, exist_ok=True)
            storage_axes = tuple(
                sorted(range(value.ndim), key=lambda i: abs(value.strides[i]), reverse=True)
            )
            logical_axes = tuple(storage_axes.index(i) for i in range(value.ndim))
            shape = value.transpose(storage_axes).shape
            mean = np.lib.format.open_memmap(
                directory / f"{name}-mean.npy", mode="w+", dtype=np.float32, shape=shape
            ).transpose(logical_axes)
            squared_difference = None
            if not self.mean_only:
                squared_difference = np.lib.format.open_memmap(
                    directory / f"{name}-m2.npy",
                    mode="w+",
                    dtype=np.float32,
                    shape=shape,
                ).transpose(logical_axes)
            self._arrays[key] = _RunningArray(mean, squared_difference, storage_axes=storage_axes)
        return self._arrays[key]

    def add(self, lead: int, products: PhysicalProducts) -> None:
        for field in fields(products):
            value = getattr(products, field.name)
            if value is not None:
                self._array(lead, field.name, value).add(value)

    def count(self, lead: int, name: str = "surface") -> int:
        return self._arrays[(lead, name)].count

    def snapshot(self, lead: int, statistic: str) -> PhysicalProducts:
        values = {}
        for field in fields(PhysicalProducts):
            running = self._arrays.get((lead, field.name))
            if running is None:
                values[field.name] = None
            elif statistic == "mean":
                values[field.name] = running.mean
            elif statistic == "standard_deviation":
                values[field.name] = running.standard_deviation()
            else:
                raise ValueError(f"Unknown statistic: {statistic}")
        return PhysicalProducts(**values)

    def reference(self, lead: int, statistic: str) -> StatisticsSnapshot:
        if statistic not in {"mean", "standard_deviation"}:
            raise ValueError(f"Unknown statistic: {statistic}")
        arrays = {}
        for field in fields(PhysicalProducts):
            running = self._arrays.get((lead, field.name))
            if running is None:
                continue
            if statistic == "standard_deviation" and running.squared_difference is None:
                raise ValueError("This accumulator tracks the mean only")
            arrays[field.name] = StatisticsArray(
                Path(running.mean.filename).resolve(),
                None
                if running.squared_difference is None
                else Path(running.squared_difference.filename).resolve(),
                running.count,
                tuple(running.storage_axes.index(i) for i in range(running.mean.ndim)),
            )
        return StatisticsSnapshot(arrays, statistic)

    def flush(self) -> None:
        for running in self._arrays.values():
            running.mean.flush()
            if running.squared_difference is not None:
                running.squared_difference.flush()

    def close(self) -> None:
        self.flush()
        self._arrays.clear()

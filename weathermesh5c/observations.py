from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone

import numpy as np

from .ingest.process import (
    _fit_bending_angle,
    _fit_refractivity,
    _load_iasi_projection,
    _nearest,
    _surface_value,
)
from .ingest.sources import MICROWAVE_STATS, SOURCES


def pack_observations(
    source: str,
    fields: Mapping[str, np.ndarray],
) -> list[tuple[datetime, np.ndarray]]:
    """Convert physical report columns to the trained hourly representation.

    Times are UTC; microwave channels start at one. IASI radiances use
    W m-2 sr-1 (m-1)-1 and the original selected-channel order. Missing
    conventional fields remain missing, not artificial zero observations.
    """
    spec = SOURCES[source]
    times = np.asarray(fields["time"], dtype="datetime64[ns]")
    # Retain the source-clock resolution used by the native training preparation.
    resolution = (
        "h" if source == "adpupa" else "m" if source in {"gpsro", "adpsfc"} else "s"
    )
    times = times.astype(f"datetime64[{resolution}]").astype("datetime64[ns]")
    count = len(times)
    if not count:
        return []
    if np.isnat(times).any():
        raise ValueError("Observation times must be finite")
    hours = times.astype("datetime64[h]") + np.timedelta64(1, "h")
    relative = (times - hours) / np.timedelta64(1, "h")

    def values(name: str, default: float = np.nan) -> np.ndarray:
        result = np.asarray(fields.get(name, np.full(count, default)), dtype=np.float64)
        if result.shape != (count,):
            raise ValueError(
                f"Observation column {name} must have one value per report"
            )
        return result

    if source == "gpsro":
        return _gpsro(fields, times, hours)
    raw = np.full((count, len(spec.columns)), np.nan, dtype=spec.dtype)

    def put(name: str, value: np.ndarray) -> None:
        raw[:, spec.columns.index(name)] = value

    lat, lon = values("lat"), values("lon")
    lon = np.where(lon > 180, lon - 360, lon)
    put("reltime_hours", relative)
    keep = np.isfinite(lat) & np.isfinite(lon)
    temperature, dewpoint, q = values("t"), values("td"), values("q")
    # Recover dew point when a standard source supplies specific humidity.
    with np.errstate(divide="ignore", invalid="ignore"):
        vapor = q * values("pres") / (0.622 + 0.378 * q)
        log_vapor = np.log(vapor / 611.2)
        derived = 243.5 * log_vapor / (17.67 - log_vapor) + 273.15
    dewpoint = np.where(np.isfinite(dewpoint), dewpoint, derived)
    if source in MICROWAVE_STATS or source == "mtiasi":
        for name, coordinate in (("lat", lat), ("lon", lon)):
            fraction, integer = np.modf(coordinate)
            put(name + "int", integer)
            put(name + "dec", fraction)
    else:
        put("lat_deg", lat)
        put("lon_deg", lon)
    if source in MICROWAVE_STATS:
        mean, std = MICROWAVE_STATS[source]
        for index in range(len(mean)):
            put(
                f"chan_{index}",
                (values(f"chan_{index + 1}") - mean[index]) / std[index],
            )
        put("said", values("satellite_id"))
        put("saza_rad", np.deg2rad(values("satellite_za")))
        put("soza_rad", np.deg2rad(values("solza")))
        keep &= (
            np.isfinite(values("satellite_id"))
            & np.isfinite(values("satellite_za"))
            & np.isfinite(values("solza"))
        )
    elif source == "mtiasi":
        radiances = np.asarray(fields["radiance"], dtype=np.float64)
        weights, bias = _load_iasi_projection()
        if radiances.shape != (count, weights.shape[0]):
            raise ValueError("IASI requires the native selected radiance channels")
        projection = np.nan_to_num(radiances, nan=0) @ weights + bias
        raw[:, spec.columns.index("pca_0") :] = projection
        # These trigonometric conventions are part of the trained representation.
        for name, field in (
            ("saza", "satellite_za"),
            ("bear", "bearing_azimuth"),
            ("soza", "solza"),
            ("solazi", "solaza"),
        ):
            put("s" + name, np.sin(values(field)))
            put("c" + name, np.cos(values(field)))
        put("ifovn", (values("field_of_view") - 1) / 120)
        keep &= (values("quality", 0) == 0) & (values("scan_line", 0) != -1)
    elif source in {"adpupa", "windborne"}:
        put("id", values("station_id", 0))
        put("pres_pa", values("pres"))
        put("unnorm_ucomp_mps", values("u"))
        put("unnorm_vcomp_mps", values("v"))
        put("unnorm_temp_k", values("t"))
        humidity = values("rh").copy()
        missing = ~np.isfinite(humidity)
        valid = missing & np.isfinite(dewpoint) & np.isfinite(temperature)
        # Keep the original model's humidity convention, including its units.
        t, td = temperature[valid], dewpoint[valid]
        if source == "windborne":
            t, td = t - 273.15, td - 273.15
        humidity[valid] = np.exp(17.625 * td / (243.04 + td)) / np.exp(
            17.625 * t / (243.04 + t)
        )
        put("unnorm_rh", humidity)
        if source == "windborne":
            put("unnorm_sh_mgpkg", q * 1e6)
        keep &= np.isfinite(values("pres"))
    elif source == "satwnd":
        put("id", values("satellite_id"))
        put("satzenith_deg", values("satellite_za"))
        put("pres_pa", values("pres"))
        put("unnorm_ucomp_mps", values("u"))
        put("unnorm_vcomp_mps", values("v"))
        keep &= np.isfinite(values("pres")) & np.isfinite(values("satellite_za"))
    elif source == "adpsfc":

        def rounded(data, digits=1):
            # Match native text formatting, including binary half-way values.
            return np.fromiter(
                (_surface_value(v, digits=digits) for v in data), dtype=np.float64
            )

        put("lat_deg", rounded(lat, 2))
        put("lon_deg", rounded(lon, 2))
        keep &= (np.abs(rounded(lat, 2)) <= 90) & (np.abs(rounded(lon, 2)) <= 180)
        if "subset" in fields:
            # Raw NCEP surface longitudes are already signed, unlike some
            # external Earth2 sources. Keep the native raw-coordinate rejection.
            keep &= np.abs(rounded(values("lon"), 2)) <= 180
        put("is_ship", values("is_ship", 0))
        put(
            "unnorm_elev_m",
            np.where(
                np.isfinite(values("station_elev")),
                np.trunc(values("station_elev")),
                99999,
            ),
        )
        put("unnorm_2d_c", rounded(dewpoint - 273.16))
        u, v = values("u"), values("v")
        if "wind_direction" in fields and "wind_speed" in fields:
            direction, speed = rounded(values("wind_direction")), rounded(
                values("wind_speed")
            )
            u, v = -speed * np.sin(np.deg2rad(direction)), -speed * np.cos(
                np.deg2rad(direction)
            )
        put("unnorm_10u_mps", u)
        put("unnorm_10v_mps", v)
        for output, field, scale, offset in (
            ("mslp_hpa", "msl", 0.01, 0),
            ("2t_c", "t", 1, -273.16),
            ("viz_m", "visibility", 1, 0),
            ("3hpr_mm", "tp:sum:3h", 1000, 0),
            ("24hpr_mm", "tp:sum:24h", 1000, 0),
        ):
            put("unnorm_" + output, rounded(values(field) * scale + offset))
    else:
        raise ValueError(f"Unsupported observation source: {source}")
    return [
        (
            datetime.fromtimestamp(
                int(hour.astype("datetime64[s]").astype(np.int64)), timezone.utc
            ),
            raw[keep & (hours == hour)],
        )
        for hour in np.unique(hours)
    ]


def _gpsro(fields, times, hours):
    spec = SOURCES["gpsro"]
    identifiers = np.asarray(fields["profile_id"])
    altitude = np.arange(0, 30_000, 100)
    result = {}
    profiles = {}
    for index, identifier in enumerate(identifiers):
        profiles.setdefault(identifier, []).append(index)
    for indices in profiles.values():
        chosen = np.asarray(indices)
        if "confidence" in fields:
            confidence = np.asarray(fields["confidence"])[chosen][0]
            if not np.isfinite(confidence) or int(confidence) == 0:
                continue
        height = np.asarray(fields["elev"])[chosen]
        r = _fit_refractivity(
            height, np.asarray(fields["gps_refractivity"])[chosen].copy()
        )
        b = _fit_bending_angle(height, np.asarray(fields["gps"])[chosen].copy())
        lat, lon = np.asarray(fields["lat"])[chosen], np.asarray(fields["lon"])[chosen]
        lon = np.where(lon > 180, lon - 360, lon)
        valid = (
            np.isfinite(height)
            & np.isfinite(r)
            & np.isfinite(b)
            & np.isfinite(lat)
            & np.isfinite(lon)
        )
        valid &= ~(
            (height >= 0) & (height <= 30_000) & ((np.abs(r) > 5) | (np.abs(b) > 4.5))
        )
        if valid.sum() < 2:
            continue
        order = np.argsort(height[valid])
        height, r, b, lat, lon = [
            np.asarray(v[valid], dtype=np.float16)[order]
            for v in (height, r, b, lat, lon)
        ]
        r = np.interp(altitude, height, r, left=np.nan, right=np.nan)
        b = np.interp(altitude, height, b, left=np.nan, right=np.nan)
        lat, lon = _nearest(altitude, height, lat), _nearest(altitude, height, lon)
        latdec, latint = np.modf(lat)
        londec, lonint = np.modf(lon)
        hour = hours[chosen][0]
        relative = float((times[chosen][0] - hour) / np.timedelta64(1, "h"))
        raw = np.stack(
            (
                np.full(len(altitude), relative),
                latint,
                latdec,
                lonint,
                londec,
                np.nan_to_num(r),
                np.nan_to_num(b),
                np.isfinite(r) & np.isfinite(b),
            ),
            axis=-1,
        ).astype(spec.dtype)
        result.setdefault(hour, []).append(raw)
    return [
        (
            datetime.fromtimestamp(
                int(hour.astype("datetime64[s]").astype(np.int64)), timezone.utc
            ),
            np.stack(result[hour]),
        )
        for hour in sorted(result)
    ]

from datetime import datetime, timezone

import numpy as np

from weathermesh5c.inference import IASI_CHANNELS
from weathermesh5c.ingest.process import process_windborne_observations
from weathermesh5c.ingest.sources import MICROWAVE_STATS, SOURCES
from weathermesh5c.observations import pack_observations


def test_microwave_physical_values_reach_native_columns():
    mean, std = MICROWAVE_STATS["1bmhs"]
    fields = {
        "time": np.array(["2026-10-28T01:12:34.500"], dtype="datetime64[ms]"),
        "lat": np.array([32.25]),
        "lon": np.array([240.25]),
        "satellite_id": np.array([223]),
        "satellite_za": np.array([30.0]),
        "solza": np.array([60.0]),
        **{f"chan_{i + 1}": np.array([mean[i] + std[i]]) for i in range(len(mean))},
    }
    [(hour, actual)] = pack_observations("1bmhs", fields)
    spec = SOURCES["1bmhs"]
    assert hour == datetime(2026, 10, 28, 2, tzinfo=timezone.utc)
    for i in range(len(mean)):
        assert actual[0, spec.columns.index(f"chan_{i}")] == 1
    assert actual[0, spec.columns.index("lonint")] == -119
    assert actual[0, spec.columns.index("londec")] == -0.75
    assert actual[0, spec.columns.index("saza_rad")] == np.float16(np.deg2rad(30))


def test_radiosonde_preserves_native_clock_and_humidity():
    fields = {
        "time": np.array(["2026-10-28T01:42:12"], dtype="datetime64[s]"),
        "lat": np.array([32.0]),
        "lon": np.array([20.0]),
        "station_id": np.array([123]),
        "pres": np.array([85000.0]),
        "t": np.array([280.0]),
        "td": np.array([274.0]),
        "rh": np.array([np.nan]),
        "u": np.array([2.25]),
        "v": np.array([-3.5]),
    }
    [(hour, actual)] = pack_observations("adpupa", fields)
    spec = SOURCES["adpupa"]
    assert hour == datetime(2026, 10, 28, 2, tzinfo=timezone.utc)
    assert actual[0, spec.columns.index("reltime_hours")] == -1
    assert actual[0, spec.columns.index("pres_pa")] == 85000
    expected = np.exp(17.625 * 274 / (243.04 + 274)) / np.exp(
        17.625 * 280 / (243.04 + 280)
    )
    assert actual[0, spec.columns.index("unnorm_rh")] == np.float32(expected)
    assert np.isnan(fields["rh"][0])


def test_surface_keeps_unrounded_wind_components_and_missing_fields():
    fields = {
        "time": np.array(["2026-10-28T01:42:12"], dtype="datetime64[s]"),
        "lat": np.array([32.0]),
        "lon": np.array([20.0]),
        "t": np.array([280.0]),
        "u": np.array([2.2567]),
        "v": np.array([-3.548]),
    }
    [(_, actual)] = pack_observations("adpsfc", fields)
    spec = SOURCES["adpsfc"]
    assert actual[0, spec.columns.index("unnorm_10u_mps")] == np.float32(2.2567)
    assert np.isnan(actual[0, spec.columns.index("unnorm_mslp_hpa")])


def test_surface_raw_geometry_uses_native_decimal_rounding():
    from weathermesh5c.ingest.process import _surface_value

    fields = {
        "time": np.array(["2026-10-28T00"], dtype="datetime64[h]"),
        "lat": np.array([2.675]),
        "lon": np.array([20.0]),
        "t": np.array([299.21]),
        "wind_direction": np.array([10.05]),
        "wind_speed": np.array([2.25]),
    }
    [(_, actual)] = pack_observations("adpsfc", fields)
    columns = SOURCES["adpsfc"].columns
    assert actual[0, columns.index("lat_deg")] == np.float32(
        _surface_value(2.675, digits=2)
    )
    expected = -_surface_value(2.25) * np.sin(np.deg2rad(_surface_value(10.05)))
    assert actual[0, columns.index("unnorm_10u_mps")] == np.float32(expected)


def test_surface_rejects_invalid_raw_coordinates_but_accepts_unsigned_provider_longitudes():
    from weathermesh5c import Inference

    fields = {
        "time": np.array(["2026-10-28T00"], dtype="datetime64[h]"),
        "lat": np.array([30.0]),
        "lon": np.array([240.0]),
        "t": np.array([280.0]),
    }
    assert len(Inference.pack_raw_observations("adpsfc", fields)) == 1
    assert (
        Inference.pack_raw_observations("adpsfc", {**fields, "subset": np.array([1])})
        == []
    )
    assert (
        Inference.pack_raw_observations("adpsfc", {**fields, "lat": np.array([90.1])})
        == []
    )


def test_raw_batch_packing_matches_full_frame(monkeypatch):
    import weathermesh5c.observations as observations
    from weathermesh5c import Inference

    generator = np.random.default_rng(1)
    weights = generator.normal(size=(len(IASI_CHANNELS), 66))
    monkeypatch.setattr(
        observations, "_load_iasi_projection", lambda: (weights, np.zeros(66))
    )
    count = 17
    fields = {
        "time": np.full(count, np.datetime64("2026-10-28T00")),
        "lat": np.full(count, 30.0),
        "lon": np.full(count, 20.0),
        "radiance": generator.normal(size=(count, len(IASI_CHANNELS))),
        **{
            name: np.zeros(count)
            for name in (
                "satellite_za",
                "bearing_azimuth",
                "solza",
                "solaza",
                "quality",
                "scan_line",
            )
        },
        "field_of_view": np.ones(count),
    }
    [(hour, expected)] = Inference.pack_raw_observations("mtiasi", fields)
    chunks = []
    for selection in (slice(0, 3), slice(3, 11), slice(11, None)):
        [(time, values)] = Inference.pack_raw_observations(
            "mtiasi", {key: value[selection] for key, value in fields.items()}
        )
        assert time == hour
        chunks.append(values)
    np.testing.assert_array_equal(np.concatenate(chunks), expected)
    [(_, merged)] = Inference.pack_raw_observations(
        "mtiasi", [fields, {key: value[:3] for key, value in fields.items()}]
    )
    np.testing.assert_array_equal(merged, expected)


def test_iasi_keeps_native_quality_and_scan_line_rejections():
    fields = {
        "time": np.array(["2026-10-28T00"] * 4, dtype="datetime64[h]"),
        "lat": np.full(4, 30.0),
        "lon": np.full(4, 20.0),
        "radiance": np.zeros((4, len(IASI_CHANNELS))),
        "quality": np.array([0, np.nan, 1, 0]),
        "scan_line": np.array([1, 2, 3, -1]),
        "satellite_za": np.zeros(4),
        "bearing_azimuth": np.zeros(4),
        "solza": np.zeros(4),
        "solaza": np.zeros(4),
        "field_of_view": np.ones(4),
    }
    [(_, actual)] = pack_observations("mtiasi", fields)
    assert len(actual) == 1
    assert np.isfinite(actual).all()


def test_gpsro_rejects_zero_confidence_before_fitting():
    fields = {
        "time": np.array(["2026-10-28T00"], dtype="datetime64[h]"),
        "profile_id": np.array(["profile"]),
        "confidence": np.array([0.0]),
    }
    assert pack_observations("gpsro", fields) == []


def test_gpsro_wraps_longitude_before_native_half_rounding():
    fields = {
        "time": np.array(["2026-10-28T00"] * 2, dtype="datetime64[h]"),
        "profile_id": np.array(["profile"] * 2),
        "lat": np.array([0.125, 0.125]),
        "lon": np.array([270.1, 270.1]),
        "elev": np.array([0.0, 1000.0]),
        "gps_refractivity": np.array([339.5, 297.0]),
        "gps": np.array([0.02839, 0.02455]),
    }
    [(_, actual)] = pack_observations("gpsro", fields)
    names = SOURCES["gpsro"].columns
    longitude = actual[..., names.index("lonint")] + actual[..., names.index("londec")]
    np.testing.assert_array_equal(longitude, np.full((1, 300), np.float16(-89.9)))


def test_windborne_physical_frame_matches_native_ingestion():
    when = datetime(2026, 10, 28, 0, 30, tzinfo=timezone.utc)
    native = process_windborne_observations(
        [
            {
                "timestamp": when.timestamp(),
                "id": 17,
                "latitude": 43,
                "longitude": -122,
                "pressure": 850,
                "temperature": 6.85,
                "humidity": 0.4,
                "specific_humidity": 3000,
                "speed_u": 2,
                "speed_v": 3,
            }
        ],
        available_at=0,
    )
    fields = {
        "time": np.array(["2026-10-28T00:30"], dtype="datetime64[m]"),
        "station_id": np.array([17]),
        "lat": np.array([43]),
        "lon": np.array([-122]),
        "pres": np.array([85000]),
        "t": np.array([280]),
        "rh": np.array([0.4]),
        "q": np.array([0.003]),
        "u": np.array([2]),
        "v": np.array([3]),
    }
    [(hour, actual)] = pack_observations("windborne", fields)
    np.testing.assert_array_equal(actual, native[hour]["x"])

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Source:
    name: str
    extension: str
    columns: tuple[str, ...]
    dtype: np.dtype


def _microwave_columns(channels: int) -> tuple[str, ...]:
    return (
        "reltime_hours",
        "said",
        "latint",
        "latdec",
        "lonint",
        "londec",
        "saza_rad",
        "soza_rad",
        *(f"chan_{index}" for index in range(channels)),
    )


SOURCES = {
    "1bamua": Source("1bamua", "bufr_d", _microwave_columns(15), np.dtype("float16")),
    "1bmhs": Source("1bmhs", "bufr_d", _microwave_columns(5), np.dtype("float16")),
    "atms": Source("atms", "bufr_d", _microwave_columns(22), np.dtype("float16")),
    "mtiasi": Source(
        "mtiasi",
        "bufr_d",
        (
            "reltime_hours",
            "latint",
            "latdec",
            "lonint",
            "londec",
            "ssaza",
            "csaza",
            "sbear",
            "cbear",
            "ssoza",
            "csoza",
            "ssolazi",
            "csolazi",
            "ifovn",
            *(f"pca_{index}" for index in range(66)),
        ),
        np.dtype("float16"),
    ),
    "gpsro": Source(
        "gpsro",
        "bufr_d.nr",
        (
            "reltime_hours",
            "latint",
            "latdec",
            "lonint",
            "londec",
            "rint",
            "bint",
            "all_present",
        ),
        np.dtype("float16"),
    ),
    "satwnd": Source(
        "satwnd",
        "bufr_d",
        (
            "id",
            "reltime_hours",
            "lat_deg",
            "lon_deg",
            "satzenith_deg",
            "pres_pa",
            "unnorm_ucomp_mps",
            "unnorm_vcomp_mps",
        ),
        np.dtype("float32"),
    ),
    "adpupa": Source(
        "adpupa",
        "bufr_d",
        (
            "id",
            "reltime_hours",
            "lat_deg",
            "lon_deg",
            "pres_pa",
            "unnorm_ucomp_mps",
            "unnorm_vcomp_mps",
            "unnorm_temp_k",
            "unnorm_rh",
        ),
        np.dtype("float32"),
    ),
    "windborne": Source(
        "windborne",
        "csv",
        (
            "id",
            "reltime_hours",
            "lat_deg",
            "lon_deg",
            "pres_pa",
            "unnorm_ucomp_mps",
            "unnorm_vcomp_mps",
            "unnorm_temp_k",
            "unnorm_rh",
            "unnorm_sh_mgpkg",
        ),
        np.dtype("float32"),
    ),
    "adpsfc": Source(
        "adpsfc",
        "bufr_d.nr",
        (
            "is_ship",
            "reltime_hours",
            "lat_deg",
            "lon_deg",
            "unnorm_elev_m",
            "unnorm_mslp_hpa",
            "unnorm_2t_c",
            "unnorm_2d_c",
            "unnorm_10u_mps",
            "unnorm_10v_mps",
            "unnorm_viz_m",
            "unnorm_3hpr_mm",
            "unnorm_24hpr_mm",
        ),
        np.dtype("float32"),
    ),
}

PUBLIC_SOURCES = tuple(name for name in SOURCES if name != "windborne")


MICROWAVE_STATS = {
    "1bamua": (
        np.array(
            [
                208.11433,
                200.07397,
                239.66342,
                252.61078,
                246.26218,
                230.94685,
                222.31076,
                216.79262,
                210.95709,
                214.30807,
                220.71764,
                228.75981,
                240.04573,
                251.177,
                238.29852,
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                40.338806,
                41.497913,
                19.817978,
                13.5695715,
                11.055716,
                8.138229,
                7.369916,
                8.368133,
                10.854924,
                11.318849,
                12.498083,
                13.425758,
                13.70429,
                12.700977,
                27.190237,
            ],
            dtype=np.float32,
        ),
    ),
    "1bmhs": (
        np.array([240.02309, 255.39546, 249.3301, 253.13164, 255.33632], dtype=np.float32),
        np.array([29.984795, 31.28215, 13.959212, 19.019217, 25.365969], dtype=np.float32),
    ),
    "atms": (
        np.array(
            [
                209.518865,
                201.116778,
                239.50126,
                249.30659,
                252.9082,
                246.775408,
                231.462015,
                222.500808,
                217.218617,
                211.59793,
                215.116865,
                221.556385,
                228.964267,
                241.159455,
                250.80906,
                237.884928,
                258.607035,
                260.176345,
                258.14215,
                255.904132,
                252.182983,
                246.94686,
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                41.237975,
                41.979399,
                21.9403,
                17.243942,
                13.995348,
                11.227189,
                8.208647,
                7.063055,
                8.130123,
                10.969226,
                11.324154,
                12.453338,
                13.527296,
                14.045363,
                13.322912,
                28.550902,
                29.529412,
                21.900062,
                17.83456,
                14.576401,
                11.84562,
                10.112091,
            ],
            dtype=np.float32,
        ),
    ),
}

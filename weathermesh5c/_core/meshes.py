import base64
import hashlib
import json

import numpy as np
import torch

from .utils import (
    MICROWAVE_N_CHANNELS,
    core_pressure_vars,
    core_sfc_vars,
    levels_full,
    load_state_norm,
)


class Mesh:
    def __init__(self):
        self.subsamp = 1
        self.resolution = 0.25

    def finalize(self):
        self.set_string_id()

    def apply_options(self, options):
        for name, value in options.items():
            if not hasattr(self, name):
                raise TypeError(f"Unknown {self.__class__.__name__} option: {name}")
            setattr(self, name, value)

    def set_string_id(self):
        raise NotImplementedError


class DAMesh(Mesh):
    def __init__(self):
        super().__init__()
        self.da_window_size = (-3, 5)
        self.encoder_subset = np.inf
        self.observation_geometry = "3d_pr"


class SurfaceData(DAMesh):
    def __init__(self, **kwargs):
        super().__init__()
        self.observation_geometry = "2d_sfc"
        self.disk_varlist = [
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
        ]
        self.vars = ["is_ship", "elev", "mslp", "2t", "2d", "10u", "10v"]
        self.var_indicators = [v + "_present?" for v in self.vars]
        self.indices = ["reltime_hours", "lat_deg", "lon_deg"]
        self.addl = [
            "sin(lat_deg)",
            "cos(lat_deg)",
            "sin(lon_deg)",
            "cos(lon_deg)",
            "sin(time_of_year)",
            "cos(time_of_year)",
            "sin(time_of_day)",
            "cos(time_of_day)",
            "solar_elevation",
        ]

        self.full_varlist = self.indices + self.vars + self.var_indicators + self.addl

        self.apply_options(kwargs)

        super().finalize()

    def shape(self):
        return [-1, len(self.full_varlist)]

    def set_string_id(self):
        self.string_id = "sfc"
        self.mesh_id = f"{self.__class__.__name__}+sfc"


class WindborneData(DAMesh):
    def __init__(self, **kwargs):
        super().__init__()
        self.disk_varlist = [
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
        ]
        self.vars = ["ucomp", "vcomp", "temp", "rh", "sh"]
        self.var_indicators = [v + "_present?" for v in self.vars]
        self.indices = ["reltime_hours", "pres_hpa", "lat_deg", "lon_deg"]
        self.addl = [
            "sin(lat_deg)",
            "cos(lat_deg)",
            "sin(lon_deg)",
            "cos(lon_deg)",
            "sin(time_of_year)",
            "cos(time_of_year)",
            "sin(time_of_day)",
            "cos(time_of_day)",
            "solar_elevation",
        ]
        self.full_varlist = self.indices + self.vars + self.var_indicators + self.addl
        self.apply_options(kwargs)
        super().finalize()

    def set_string_id(self):
        self.string_id = "windborne"
        self.mesh_id = f"{self.__class__.__name__}+windborne"

    def shape(self):
        return [-1, len(self.full_varlist)]


class RadiosondeData(DAMesh):
    def __init__(self, **kwargs):
        super().__init__()
        self.disk_varlist = [
            "id",
            "reltime_hours",
            "lat_deg",
            "lon_deg",
            "pres_pa",
            "unnorm_ucomp_mps",
            "unnorm_vcomp_mps",
            "unnorm_temp_k",
            "unnorm_rh",
        ]
        self.vars = ["ucomp", "vcomp", "temp", "rh"]
        self.var_indicators = [v + "_present?" for v in self.vars]
        self.indices = ["reltime_hours", "pres_hpa", "lat_deg", "lon_deg"]
        self.addl = [
            "sin(lat_deg)",
            "cos(lat_deg)",
            "sin(lon_deg)",
            "cos(lon_deg)",
            "sin(time_of_year)",
            "cos(time_of_year)",
            "sin(time_of_day)",
            "cos(time_of_day)",
            "solar_elevation",
        ]
        self.full_varlist = self.indices + self.vars + self.var_indicators + self.addl
        self.encoder_subset = 100_000
        self.apply_options(kwargs)
        super().finalize()

    def set_string_id(self):
        self.string_id = "radiosonde"
        self.mesh_id = f"{self.__class__.__name__}+radiosonde"

    def shape(self):
        return [-1, len(self.full_varlist)]


class SatwindData(DAMesh):
    def __init__(self, **kwargs):
        super().__init__()
        self.disk_varlist = [
            "id",
            "reltime_hours",
            "lat_deg",
            "lon_deg",
            "satzenith_deg",
            "pres_pa",
            "unnorm_ucomp_mps",
            "unnorm_vcomp_mps",
        ]
        self.vars = ["ucomp", "vcomp"]
        self.var_indicators = [v + "_present?" for v in self.vars]
        self.indices = ["reltime_hours", "pres_hpa", "lat_deg", "lon_deg"]
        self.addl = [
            "sin(lat_deg)",
            "cos(lat_deg)",
            "sin(lon_deg)",
            "cos(lon_deg)",
            "sin(satzenith_deg)",
            "cos(satzenith_deg)",
        ]
        self.full_varlist = self.indices + self.vars + self.var_indicators + self.addl
        self.encoder_subset = 200_000
        self.apply_options(kwargs)
        super().finalize()

    def set_string_id(self):
        self.string_id = "satwnd"
        self.mesh_id = f"{self.__class__.__name__}+satwind"

    def shape(self):
        return [-1, len(self.full_varlist)]


class MicrowaveData(DAMesh):
    def __init__(self, instrument, **kwargs):
        super().__init__()
        self.instrument = instrument
        self.observation_geometry = "2d_pr+sfc"
        self.nchan = MICROWAVE_N_CHANNELS[instrument]
        self.sats = self._satellite_ids(instrument)
        self.vars = ["chan_" + str(i) for i in range(self.nchan)]
        self.var_indicators = [v + "_present?" for v in self.vars]

        self.disk_varlist = [
            "reltime_hours",
            "said",
            "latint",
            "latdec",
            "lonint",
            "londec",
            "saza_rad",
            "soza_rad",
        ] + ["chan_" + str(i) for i in range(self.nchan)]
        self.indices = ["reltime_hours", "lat_deg", "lon_deg"]
        self.addl = [
            "sin(lat_deg)",
            "cos(lat_deg)",
            "sin(lon_deg)",
            "cos(lon_deg)",
            "sin(saza_rad)",
            "cos(saza_rad)",
            "sin(soza_rad)",
            "cos(soza_rad)",
        ] + ["onehot_said_" + str(i) for i in range(len(self.sats))]
        self.full_varlist = self.indices + self.vars + self.var_indicators + self.addl
        self.encoder_input_dtype = torch.float16
        self.encoder_subset = 200_000
        super().finalize()

        self.apply_options(kwargs)

    def set_string_id(self):
        self.string_id = self.instrument
        self.mesh_id = f"{self.__class__.__name__}+{self.instrument}"

    @staticmethod
    def _satellite_ids(instrument):
        return {
            "atms": [224, 225, 226],
            "1bamua": [3, 4, 5, 206, 207, 209, 223, 784],
            "1bmhs": [3, 4, 5, 209, 223],
        }[instrument]

    def shape(self):
        return [-1, len(self.full_varlist)]  # lat lon hack, gets stripped away


class GPSROData(DAMesh):
    def __init__(self, **kwargs):
        super().__init__()
        self.observation_geometry = "3d_pr"
        self.nchan = 300

        self.disk_varlist = [
            "reltime_hours",
            "latint",
            "latdec",
            "lonint",
            "londec",
            "rint",
            "bint",
            "all_present",
        ]
        self.indices = ["reltime_hours", "lat_deg", "lon_deg"]
        self.vars = ["rint", "bint"]
        self.var_indicators = [v + "_present?" for v in self.vars]
        self.addl = ["sin(lat_deg)", "cos(lat_deg)", "sin(lon_deg)", "cos(lon_deg)"]
        self.full_varlist = self.indices + self.vars + self.var_indicators + self.addl

        self.encoder_input_dtype = torch.float16
        self.encoder_subset = 200_000
        self.apply_options(kwargs)
        super().finalize()

    def set_string_id(self):
        self.string_id = "gpsro"
        self.mesh_id = "GPSROData+gpsro"

    def shape(self):
        return [-1, self.nchan, len(self.full_varlist)]


class IASIPcaData(DAMesh):
    def __init__(self, pca=66, **kwargs):
        super().__init__()
        self.observation_geometry = "2d_pr+sfc"
        self.npca = pca
        self.pca_vars = [f"pca_{i}" for i in range(self.npca)]
        self.vars = [
            "ssaza",
            "csaza",
            "sbear",
            "cbear",
            "ssoza",
            "csoza",
            "ssolazi",
            "csolazi",
            "ifovn",
        ] + self.pca_vars
        self.disk_varlist = ["reltime_hours", "latint", "latdec", "lonint", "londec"] + self.vars

        self.indices = ["reltime_hours", "lat_deg", "lon_deg"]
        self.addl = ["sin(lat_deg)", "cos(lat_deg)", "sin(lon_deg)", "cos(lon_deg)"]
        self.full_varlist = self.indices + self.vars + self.addl
        self.encoder_input_dtype = torch.float32
        self.encoder_subset = 200_000
        self.apply_options(kwargs)
        super().finalize()

    def set_string_id(self):
        self.string_id = "iasi_pca"
        self.mesh_id = f"{self.__class__.__name__}+iasi__{self.npca}"

    def shape(self):
        return [-1, len(self.full_varlist)]  # lat lon hack, gets stripped away


def get_grid_latlons(res):
    lats = np.arange(90, -90.01, -res)[:-1]

    lons = np.arange(0, 359.99, res)
    lons[lons >= 180] -= 360

    return lats, lons


class LatLonGrid(Mesh):
    def __init__(
        self,
        *,
        source,
        extra_sfc_vars,
        input_levels,
        levels,
        extra_sfc_pad=0,
        pr_quantile_vars=(),
        sfc_quantile_vars=(),
    ):
        super().__init__()
        self.source = source
        self.extra_sfc_vars = extra_sfc_vars
        self.extra_sfc_pad = extra_sfc_pad
        self.input_levels = input_levels
        self.levels = levels
        self.pr_quantile_vars = pr_quantile_vars
        self.sfc_quantile_vars = sfc_quantile_vars
        self.lats, self.lons = get_grid_latlons(self.resolution)
        self._set_variables()
        self._set_grid()
        super().finalize()

    def shape(self):
        return torch.Size([len(self.lats), len(self.lons), self.n_vars])

    def set_string_id(self):
        digest = self._identity_digest()
        self.string_id = f"{self.source}>{self.n_levels}p{self.n_pr_vars}s{self.n_sfc_vars - self.extra_sfc_pad}z{self.extra_sfc_pad}r{self.res}-{digest}"
        self.mesh_id = f"{self.__class__.__name__}+{self.source}__{len(self.input_levels)}->{len(self.levels)}__{self.n_pr_vars}+{self.n_sfc_vars}+{self.extra_sfc_pad}+{len(self.pr_quantile_vars)}+{len(self.sfc_quantile_vars)}"

    def _set_variables(self):
        expected_levels = int(self.source.split("-")[1])
        if len(self.input_levels) != expected_levels:
            raise ValueError(
                f"{self.source} needs {expected_levels} input levels, got {len(self.input_levels)}"
            )
        self.pressure_vars = core_pressure_vars
        self.sfc_vars = core_sfc_vars + self.extra_sfc_vars + ["zeropad"] * self.extra_sfc_pad
        self.full_varlist = [
            f"{variable}_{level}" for variable in self.pressure_vars for level in self.levels
        ] + self.sfc_vars

        self.n_levels = len(self.levels)
        self.wh_lev = [levels_full.index(level) for level in self.levels]
        self.n_pr_vars = len(self.pressure_vars)
        self.n_pr = self.n_levels * self.n_pr_vars
        self.n_sfc = len(self.sfc_vars)
        self.n_sfc_vars = self.n_sfc
        self.n_vars = self.n_pr + self.n_sfc

    def _set_grid(self):
        longitudes, latitudes = np.meshgrid(self.lons, self.lats)
        self.res = self.resolution * self.subsamp
        self.xpos = np.stack((latitudes / 90, longitudes / 180), axis=2)
        self.state_norm_stds, self.state_norm_means = load_state_norm(self)

    def _identity_digest(self):
        identity = {
            "mesh_type": self.__class__.__name__,
            "pressure_vars": self.pressure_vars,
            "sfc_vars": self.sfc_vars,
            "full_varlist": self.full_varlist,
            "levels": self.levels,
            "lats": self.lats.tolist(),
            "lons": self.lons.tolist(),
            "res": self.res,
        }
        serialized = json.dumps(identity, indent=2)
        digest = base64.b64encode(hashlib.sha256(serialized.encode()).digest()[:3]).decode()
        return digest.replace("+", "a").replace("/", "b")

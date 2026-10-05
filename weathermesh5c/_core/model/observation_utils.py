import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn


class ResidualMLP(nn.Module):
    def __init__(self, dim, out_dim, act_layer=nn.GELU, final_act=False, inner_dim=None):
        super().__init__()
        if inner_dim is None:
            inner_dim = dim if dim > out_dim else out_dim
        self.shortcut = nn.Linear(dim, out_dim)
        self.fc1 = nn.Linear(dim, inner_dim)
        self.fc2 = nn.Linear(inner_dim, inner_dim)
        self.fc3 = nn.Linear(inner_dim, out_dim)
        self.act = act_layer()
        self.final_act = final_act

    def forward(self, x):
        xsave = x.clone()
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        if self.final_act:
            x = self.act(x)
        x = x + self.shortcut(xsave)
        return x


def normalize_observations(x, mesh):
    if "lat_deg" in mesh.full_varlist:
        x[:, mesh.full_varlist.index("lat_deg")] /= 90.0
    if "lon_deg" in mesh.full_varlist:
        x[:, mesh.full_varlist.index("lon_deg")] /= 180.0
    if "pres_hpa" in mesh.full_varlist:
        x[:, mesh.full_varlist.index("pres_hpa")] /= 1000.0
    minimum, maximum = mesh.da_window_size
    max_window = np.max([np.abs(minimum), np.abs(maximum)])
    if "reltime_hours" in mesh.full_varlist:
        x[:, mesh.full_varlist.index("reltime_hours")] /= max_window
    return x


def observation_latent_shape(config, observation_geometry, hlmesh):
    if observation_geometry == "3d_pr":
        D = len(config.latent_levels)
        C = config.assimilation_latent_size
    elif observation_geometry == "2d_pr+sfc":
        D = 1
        C = config.assimilation_latent_size * (1 + len(config.latent_levels))
    elif observation_geometry == "2d_sfc":
        D = 1
        C = config.assimilation_latent_size
    else:
        raise ValueError(f"observation_geometry {observation_geometry} not supported")

    B = 1
    Nvert = int(hlmesh.vertices.shape[0])
    return B, D, Nvert, C


def flat_observation_latent_shape(config, observation_geometry, hlmesh):
    if observation_geometry == "3d_pr":
        D = len(config.latent_levels)
        C = config.assimilation_latent_size
    elif observation_geometry == "2d_pr+sfc":
        D = len(config.latent_levels) + 1
        C = config.assimilation_latent_size
    elif observation_geometry == "2d_sfc":
        D = 1
        C = config.assimilation_latent_size
    else:
        raise ValueError(f"observation_geometry {observation_geometry} not supported")

    B = 1
    Nvert = int(hlmesh.vertices.shape[0])
    return B, D, Nvert, C


def global_latent_shape(config, hlmesh):
    B = 1
    D = len(config.latent_levels) + 1
    Nvert = int(hlmesh.vertices.shape[0])
    C = config.assimilation_latent_size

    return B, D, Nvert, C


def expand_observation_latent(latent, config, observation_geometry, hlmesh):
    B, D, Nvert, C = global_latent_shape(config, hlmesh)
    assert latent.shape == observation_latent_shape(config, observation_geometry, hlmesh)

    if observation_geometry == "3d_pr":
        assert latent.shape == (B, D - 1, Nvert, C)
    elif observation_geometry == "2d_pr+sfc":
        assert latent.shape == (B, 1, Nvert, D * C)

        latent = latent.view(B, 1, Nvert, D, C).permute(0, 3, 2, 4, 1).squeeze(-1)
    elif observation_geometry == "2d_sfc":
        assert latent.shape == (B, 1, Nvert, C)
    else:
        raise ValueError(f"observation_geometry {observation_geometry} not supported")

    return latent


def locate_horizontal(lat, lon, hlmesh, device):

    lat = lat.float()
    lon = lon.float()
    cpulat = lat.cpu().numpy()
    cpulon = lon.cpu().numpy()

    ob_vectors = np.column_stack(
        [
            np.cos(np.radians(cpulat)) * np.cos(np.radians(cpulon)),
            np.cos(np.radians(cpulat)) * np.sin(np.radians(cpulon)),
            np.sin(np.radians(cpulat)),
        ]
    )

    if not hasattr(hlmesh, "tree"):
        hlmesh.tree = cKDTree(hlmesh.vertices.cpu().numpy())

    distances, indices = hlmesh.tree.query(ob_vectors, k=1)
    distances = torch.tensor(distances, dtype=torch.float32)
    indices = torch.tensor(indices, dtype=torch.int32)
    distances = distances / torch.max(distances)
    distances = distances.to(device)

    nearest_points = hlmesh.vertices[indices].to(device=device, dtype=torch.float32)
    ob_vectors = torch.tensor(ob_vectors, dtype=torch.float32).to(device)

    lat1 = torch.arcsin(nearest_points[:, 2])
    lon1 = torch.arctan2(nearest_points[:, 1], nearest_points[:, 0])
    lat2 = torch.arcsin(ob_vectors[:, 2])  # latitude of points B
    lon2 = torch.arctan2(ob_vectors[:, 1], ob_vectors[:, 0])  # longitude of points B

    dlon = lon2 - lon1
    __y = torch.sin(dlon) * torch.cos(lat2)
    __x = torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon)
    azimuths = torch.atan2(__y, __x)

    azimuths = (azimuths + 2 * torch.pi) % (2 * torch.pi)

    sin = torch.sin(azimuths) * distances / 2 + 0.5
    cos = torch.cos(azimuths) * distances / 2 + 0.5

    return sin, cos, indices.to(torch.int64).to(device)


def locate_pressure(pressure, config, observation_geometry, device):
    if observation_geometry == "3d_pr":
        assert pressure is not None, "Pressure is required for 3d_pr"

        valid_mask = (pressure > 0) & (pressure < 1085)

        min_level = config.outputs[0].levels[0]
        max_level = config.latent_levels[-1]
        pressure = pressure.clip(min_level, max_level)

        latent_levels = torch.tensor(config.latent_levels, device=device).contiguous()
        level_inds = torch.searchsorted(latent_levels, pressure).to(torch.int64)

        expanded_latent_levels = torch.tensor(
            [min_level] + config.latent_levels, device=device
        ).contiguous()
        level_ths = (pressure - expanded_latent_levels[level_inds]) / (
            expanded_latent_levels[level_inds + 1] - expanded_latent_levels[level_inds]
        )
        return level_ths, level_inds, valid_mask
    else:
        zero_inds = torch.zeros([pressure.shape[0]], dtype=torch.int64, device=device)
        valid_mask = torch.ones(pressure.shape[0], dtype=torch.bool, device=device)
        return None, zero_inds, valid_mask


def normalize_relative_time(reltime, window_size):
    minimum, maximum = window_size
    minimum, maximum = np.abs(minimum), np.abs(maximum)
    scale = minimum + maximum
    ths = (reltime + minimum) / scale
    assert torch.all((ths >= 0) & (ths <= 1)), (
        f"All elements in ths must be between 0 and 1, got {torch.min(ths)} and {torch.max(ths)} where {minimum} and {scale} where {min(reltime)} and {max(reltime)}"
    )
    return ths


def locate_observations(
    pressure, lat, lon, reltime, observation_geometry, config, window_size, device, hlmesh
):

    ths = []
    inds = []

    pres_ths, pres_inds, valid_mask = locate_pressure(
        pressure, config, observation_geometry, device
    )
    if pres_ths is not None:
        ths.append(pres_ths)
    inds.append(pres_inds)

    sin, cos, indices = locate_horizontal(lat, lon, hlmesh, device)
    ths.append(sin)
    ths.append(cos)
    inds.append(indices)

    reltime_ths = normalize_relative_time(reltime, window_size)
    ths.append(reltime_ths)
    return inds, ths, valid_mask

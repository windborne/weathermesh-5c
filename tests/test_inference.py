from types import SimpleNamespace

import numpy as np
import pytest
import torch

import weathermesh5c.inference as inference
from weathermesh5c.ensemble import DiskStatistics
from weathermesh5c.output import (
    DecodedProducts,
    aggregate_decoded_products,
    physicalize_statistics,
)


@pytest.mark.parametrize("pressure", [False, True])
def test_decode_matches_native_single_member_statistics(tmp_path, monkeypatch, pressure):
    mesh = SimpleNamespace(
        lats=np.arange(2),
        lons=np.arange(3),
        levels=[500],
        state_norm_means=torch.arange(17).float(),
        state_norm_stds=torch.arange(1, 18).float(),
    )
    model = torch.nn.Module()
    model.anl_encoders = model.decoders = [SimpleNamespace(mesh=mesh)]
    model.obs_encoders = []
    runtime = inference.Inference(model)
    surface = torch.arange(72).reshape(1, 2, 3, 12).float() / 10
    surface[..., 9] = -0.0
    levels = torch.arange(30).reshape(1, 2, 3, 5).float() / 10
    levels[..., 0] = -0.0

    def decode(*args, include_pressure):
        return DecodedProducts(surface.clone(), levels.clone() if include_pressure else None)

    monkeypatch.setattr(inference, "decode_products_for_transfer", decode)
    statistics = DiskStatistics(tmp_path)
    statistics.add(
        0, aggregate_decoded_products(decode(include_pressure=pressure), runtime._transform)
    )
    reference = physicalize_statistics(statistics.snapshot(0, "mean"), runtime._transform, "mean")
    reference.surface[9] /= np.float32(1000)
    expected = (
        np.concatenate((reference.pressure.reshape(5, 2, 3), reference.surface))
        if pressure
        else reference.surface
    )
    actual = runtime.decode(torch.zeros(2, 1), pressure=pressure)
    for member in actual:
        np.testing.assert_array_equal(member.view(np.uint32), expected.view(np.uint32))
    statistics.close()

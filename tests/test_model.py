from datetime import datetime, timezone
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from weathermesh5c._core.model import data
from weathermesh5c._core.model.assimilation import AssimilationTransformer, WeatherMeshModel
from weathermesh5c._core.model.heal import LatentMesh
from weathermesh5c._core.model.observation_encoder import ENCODER_CONFIG_GPSRO_MAIN
from weathermesh5c._core.triconatten import tri_conatten


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("heads", [2, 3])
def test_neighborhood_attention_matches_dense_masked_attention(heads):
    generator = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = [
        torch.randn((4, 4, heads, 32), generator=generator, device="cuda", dtype=torch.float16)
        for _ in range(3)
    ]
    q = q / 10
    k = k / 10
    which = torch.arange(16, device="cuda", dtype=torch.int32)[None]
    which[:, -1] = -1
    mask = torch.ones((16, 16), device="cuda", dtype=torch.bool).tril()
    mask[:, -1] = False
    queries, keys, values = (x.reshape(16, heads, 32).transpose(0, 1).float() for x in (q, k, v))
    scores = queries @ keys.transpose(-1, -2)
    expected = (scores.masked_fill(~mask, -torch.inf).softmax(-1) @ values).transpose(0, 1)

    actual = tri_conatten(q, k, v, which, mask).reshape(16, heads, 32)

    torch.testing.assert_close(actual.float(), expected, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("vertices", [1, 4])
@pytest.mark.parametrize("mix", ["observations", "vertical"])
def test_assimilation_mixing_retains_independent_contiguous_state(vertices, mix):
    combined = []

    def transform(values, transformer):
        result = values * 2
        combined.append(result)
        return result

    model = SimpleNamespace(
        _matching_levels=AssimilationTransformer._matching_levels,
        _transform_chunks=transform,
        tr0ds={"test": None},
        tr1ds=[None],
    )
    state = {
        "analysis": torch.full((1, vertices, 2, 3), 1, dtype=torch.float16),
        "scratch": torch.full((1, vertices, 2, 3), 2, dtype=torch.float16),
        "observations": [torch.full((1, vertices, 1, 3), 3, dtype=torch.float16)],
        "analysis_levels": [[0, 1]],
        "scratch_levels": [[0, 1]],
        "observation_levels": [[1]],
    }
    if mix == "observations":
        result = AssimilationTransformer._mix_observations(model, state, ["test"])
        outputs = result["observations"]
        expected = [6]
    else:
        result = AssimilationTransformer._mix_vertical(model, state, 0)
        outputs = [result["analysis"], result["scratch"], *result["observations"]]
        expected = [2, 4, 6]
    original = combined[0].clone()
    for output, value in zip(outputs, expected, strict=True):
        assert output.is_contiguous()
        assert output.dtype == torch.float16
        assert torch.all(output == value)
        output.add_(100)
    assert torch.equal(combined[0], original)


@pytest.mark.parametrize("steps, expected", [(2, 26), (3, 42)])
@pytest.mark.parametrize("explicit_hour", [False, True])
def test_assimilation_advances_between_updates_and_retains_scratch(
    steps, expected, explicit_hour
):
    events = []

    class Encoder:
        mesh = SimpleNamespace(mesh_id="background")

        def __call__(self, values, timestamp):
            events.append(("encode", timestamp.item()))
            return values

    def advance(state, hours, start_hour):
        events.append(("advance", hours, start_hour))
        return state + 10

    def assimilate(state, ids):
        observation = state["observations"][0]
        events.append(("assimilate", observation))
        state["analysis"] = state["analysis"] + observation + state["scratch"]
        state["scratch"] = state["scratch"] + 1
        return state

    model = SimpleNamespace(
        config=SimpleNamespace(
            assimilation_hours=[-999 if explicit_hour else -3 * (steps - 1)],
            latent_levels=[],
        ),
        anl_encoders=[Encoder()],
        scratch_space=torch.ones(1, 1),
        background_tokens_per_level=1,
        step_hours=3,
        _advance=advance,
        _encode_observations=lambda cycle: ([cycle["obs"]], [], []),
        da_transformer=assimilate,
    )
    cycles = {
        -3 * steps: {
            "has_analysis": True,
            "background": torch.zeros(1, 1, 1, 1),
            "timestamp": torch.tensor(100),
        }
    }
    for index, hour in enumerate(range(-3 * (steps - 1), 1, 3), 1):
        cycles[hour] = {"has_analysis": False, "has_observations": True, "obs": index}
    kwargs = {"first_hour": -3 * (steps - 1)} if explicit_hour else {}
    result, offset = WeatherMeshModel.assimilate(model, cycles, **kwargs)
    assert result.item() == expected
    assert offset == -3 * steps
    assert events == [("encode", 100)] + [
        event
        for index in range(1, steps + 1)
        for event in (("advance", 3, (steps - index) * 3), ("assimilate", index))
    ]


@pytest.mark.parametrize(
    ("dim", "dtype", "expected"),
    [
        (32, torch.float16, "f20f9cc805c7273487af2c666ba196d45f1b8d9cd37a9837fe0eb9d7048d50d4"),
        (32, torch.float32, "f14995c688c858122ea69a5983d997a69fc23cb1f71c253396242b64e5a2e366"),
        (35, torch.float16, "c70bd196bda7c3692b1110fe137cc33ea3ee3e2dcfb9486e974f90a410ef4a9a"),
        (35, torch.float32, "b0b99c2ba011918683f30cc189b33e644b8d0bcfbb537d6106b401b5d5072d8d"),
    ],
)
def test_position_embeddings_preserve_reference_values(dim, dtype, expected):
    coordinates = torch.tensor([[0.0, 0.0], [180.0, 45.0], [-90.0, -60.0]])
    embedding = LatentMesh.posemb_mesh(None, coordinates, 6, dim, dtype=dtype)
    assert embedding.shape == (3, 6, dim)
    assert sha256(embedding.numpy().tobytes()).hexdigest() == expected


def test_gpsro_profile_shape_matches_checkpoint() -> None:
    assert ENCODER_CONFIG_GPSRO_MAIN.profile_latent_size == 128
    assert ENCODER_CONFIG_GPSRO_MAIN.profile_group_size == 12


@pytest.mark.parametrize(
    ("generator", "args", "expected"),
    (
        (
            data._solar_altitude,
            (0, 0),
            "e7a39715f1aff4cae475380c5d2cbbe9abf7ce5df06fa3f874399c1545be7481",
        ),
        (
            data._solar_altitude,
            (183, 0),
            "5cb681dd0534f9ebf9ef54e3859959fb362d0deddae4ad51130a26f5ec87ccc7",
        ),
        (
            data._solar_altitude,
            (365, 23),
            "64de708b3675ea332aa711964a98756265da6640f6f51820719954f2da5cc9a5",
        ),
        (
            data._direct_radiation,
            (24,),
            "00a3411a6edf81441582d054214e178e7bcc8bb1d96ecec7390726a1da2c93f4",
        ),
    ),
)
def test_generated_fields_match_production(generator, args, expected: str) -> None:
    values = generator(*args)

    assert values.shape == (720, 1440)
    assert values.dtype == np.float32
    assert sha256(values.tobytes()).hexdigest() == expected


def test_additional_variables_preserve_trained_time_conventions() -> None:
    when = datetime(2024, 12, 31, 23, tzinfo=timezone.utc)
    timestamps = torch.tensor([int(when.timestamp())])

    values = data.get_additional_vars(timestamps)

    assert values.shape == (1, 720, 1440, 4)
    assert values.dtype == torch.float16
    assert (
        sha256(values.numpy().tobytes()).hexdigest()
        == "d32ad0e9987fa942a5b87b4fa72fb4e77599c3e223e3f68a7ea4692bb59be800"
    )

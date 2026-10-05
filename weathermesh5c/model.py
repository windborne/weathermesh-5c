from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .assets import RuntimeBundle


def build_model(bundle: str | Path | RuntimeBundle):
    bundle = bundle if isinstance(bundle, RuntimeBundle) else RuntimeBundle(bundle)
    from ._core.meshes import (
        GPSROData,
        IASIPcaData,
        LatLonGrid,
        MicrowaveData,
        RadiosondeData,
        SatwindData,
        SurfaceData,
        WindborneData,
    )
    from ._core.model.assimilation import (
        AssimilationConfig,
        AssimilationTransformer,
        WeatherMeshModel,
    )
    from ._core.model.config import ModelConfig
    from ._core.model.decoder import (
        ForecastPredecoder,
        GlobalForecastDecoder,
    )
    from ._core.model.encoder import GlobalEncoder
    from ._core.model.forker import EnsembleForker
    from ._core.model.heal import EncoderMesh, LatentMesh
    from ._core.model.healpix_transformer import HealpixProcessor
    from ._core.model.observation_encoder import (
        ENCODER_CONFIG_GPSRO_MAIN,
        ENCODER_CONFIG_MAIN,
        GPSROEncoder,
        ObservationEncoder,
    )
    from ._core.utils import (
        core_pressure_vars,
        core_sfc_vars,
        get_quantile_vars,
        levels_ecm1,
        levels_joank,
        levels_medium,
    )

    extra_input = ["168_2d", "246_100u", "247_100v", "034_sstk"]
    extra_output = [
        "168_2d",
        "246_100u",
        "247_100v",
        "15_msnswrf",
        "45_tcc",
        "ntp-3h",
        "mx2t-3h",
        "mn2t-3h",
    ]
    extra_sfc_pad = len(extra_output) - len(extra_input)

    n_quantiles = 30
    pr_quantile_vars = get_quantile_vars(core_pressure_vars, n_quantiles)
    sfc_quantile_vars = get_quantile_vars(core_sfc_vars + extra_output, n_quantiles)

    input_mesh = LatLonGrid(
        source="ens_rt-16",
        extra_sfc_vars=extra_input,
        extra_sfc_pad=extra_sfc_pad,
        input_levels=levels_ecm1,
        levels=levels_joank,
    )
    output_mesh = LatLonGrid(
        source="era5-28",
        extra_sfc_vars=extra_output,
        input_levels=levels_medium,
        levels=levels_joank,
        pr_quantile_vars=pr_quantile_vars,
        sfc_quantile_vars=sfc_quantile_vars,
    )

    observation_limit = float("inf")
    observation_window = (-1.5, 1.5)
    mesh_1bmhs = MicrowaveData(
        "1bmhs", encoder_subset=observation_limit, da_window_size=observation_window
    )
    mesh_1bamua = MicrowaveData(
        "1bamua", encoder_subset=observation_limit, da_window_size=observation_window
    )
    mesh_atms = MicrowaveData(
        "atms", encoder_subset=observation_limit, da_window_size=observation_window
    )
    mesh_gpsro = GPSROData(encoder_subset=observation_limit, da_window_size=observation_window)
    mesh_iasi = IASIPcaData(encoder_subset=observation_limit, da_window_size=observation_window)
    mesh_adpupa = RadiosondeData(
        encoder_subset=observation_limit, da_window_size=observation_window
    )
    mesh_satwnd = SatwindData(encoder_subset=observation_limit, da_window_size=observation_window)
    mesh_windborne = WindborneData(
        encoder_subset=observation_limit, da_window_size=observation_window
    )
    mesh_surface = SurfaceData(encoder_subset=observation_limit, da_window_size=observation_window)

    config = ModelConfig(
        bundle=bundle,
        input_mesh=input_mesh,
        output_mesh=output_mesh,
        latent_size=1024,
        pressure_depth=6,
        encoder_decoder_depth=2,
        patch_size=(5, 8, 8),
        decoder_chunks=40,
        decoder_latent_size=192,
        decoder_mlp_size=1024,
        fork_depth=4,
        noise_projection_scale=24,
        denoising_hours=12,
    )

    heal_mesh = EncoderMesh(8, bundle)
    latent_mesh = LatentMesh(
        depth=5,
        D=6,
        KL=21,
        KD=5,
        B=16,
        dim=config.latent_size,
        BSh=16,
        tri=True,
        bundle=bundle,
    )
    config.fork_hlmesh = latent_mesh
    config.hmesh = heal_mesh
    config.hlmesh = latent_mesh
    forker = EnsembleForker(config)
    encoder = GlobalEncoder(input_mesh, config)
    processor = HealpixProcessor(
        dim=config.latent_size,
        depth=config.pr_depth,
        num_heads=config.num_heads,
        hlmesh=latent_mesh,
    )
    global_decoder = GlobalForecastDecoder(
        output_mesh,
        config,
        n_quantiles=n_quantiles,
        pr_quantile_vars=pr_quantile_vars,
        sfc_quantile_vars=sfc_quantile_vars,
    )
    predecoder = ForecastPredecoder(config)

    da_config = AssimilationConfig(
        assimilation_depth=8,
        assimilation_latent_size=1024,
        observation_mix_repeats=2,
        observation_mix_depth=1,
        observation_attention_size=1024,
        observation_feedforward_size=4096,
        observation_head_size=32,
        vertical_mix_depth=1,
        vertical_attention_size=1024,
        vertical_feedforward_size=4096,
        vertical_head_size=32,
        spatial_attention_size=1024,
        spatial_heads=32,
        spatial_mlp_ratio=4,
    )
    config.apply_assimilation(da_config)

    microwave_config = replace(ENCODER_CONFIG_MAIN, background_tokens=6)
    windborne_config = replace(ENCODER_CONFIG_MAIN, latent_size=1024)

    model = WeatherMeshModel(
        config,
        analysis_encoders=[encoder],
        observation_encoders=[
            ObservationEncoder([mesh_1bmhs, mesh_1bamua, mesh_atms], config, microwave_config),
            ObservationEncoder([mesh_iasi], config, microwave_config),
            GPSROEncoder([mesh_gpsro], config, ENCODER_CONFIG_GPSRO_MAIN),
            ObservationEncoder([mesh_adpupa, mesh_windborne], config, windborne_config),
            ObservationEncoder([mesh_satwnd], config, ENCODER_CONFIG_MAIN),
            ObservationEncoder([mesh_surface], config, ENCODER_CONFIG_MAIN),
        ],
        da_transformer=AssimilationTransformer(config),
        predecoder=predecoder,
        decoders=[global_decoder],
        processors={"3": processor},
        forker=forker,
    )

    assert len(model.state_dict()) == 984, "WeatherMesh-5c state-dictionary contract changed"
    return model

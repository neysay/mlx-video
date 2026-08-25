"""The combined vocoder+BWE checkpoint's config handling.

2.5-era model directories nest the full architecture in config.json;
2.3-era directories ship the SAME checkpoint under a marker config with no
architecture at all. Building the marker case from dataclass defaults made
a module whose weights silently failed to load (strict=False) and whose
forward died at conv_post — 48 channels into a 24-channel conv."""

from mlx_video.models.ltx_2.audio_vae.vocoder import (
    _COMBINED_BWE_ARCH,
    _COMBINED_VOCODER_ARCH,
    _combined_configs,
)


def test_marker_config_gets_the_combined_architecture():
    vocoder_cfg, bwe_cfg = _combined_configs(
        {"type": "bigvgan", "has_bwe_generator": True}
    )
    assert vocoder_cfg is _COMBINED_VOCODER_ARCH
    assert bwe_cfg is _COMBINED_BWE_ARCH


def test_nested_configs_pass_through_untouched():
    nested = {"vocoder": {"upsample_initial_channel": 1536}, "bwe": {"hop_length": 80}}
    vocoder_cfg, bwe_cfg = _combined_configs(nested)
    assert vocoder_cfg is nested["vocoder"]
    assert bwe_cfg is nested["bwe"]


def test_architecture_matches_the_shipped_checkpoint_schedule():
    """Channels halve per upsample stage, so conv_post's in-channels are
    initial // 2**stages. The shipped checkpoint carries conv_post weights
    of 24 (base) and 16 (bwe) in-channels — the schedule must land there."""
    base = _COMBINED_VOCODER_ARCH
    bwe = _COMBINED_BWE_ARCH
    assert base["upsample_initial_channel"] // 2 ** len(base["upsample_rates"]) == 24
    assert bwe["upsample_initial_channel"] // 2 ** len(bwe["upsample_rates"]) == 16
    assert len(base["upsample_rates"]) == len(base["upsample_kernel_sizes"])
    assert len(bwe["upsample_rates"]) == len(bwe["upsample_kernel_sizes"])

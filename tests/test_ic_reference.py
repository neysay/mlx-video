"""In-context reference conditioning (IC-LoRA) -- positions, append, slice."""

import json
import struct

import mlx.core as mx
import numpy as np
import pytest

from mlx_video.models.ltx_2.conditioning.latent import (
    ReferenceContext,
    VideoConditionByReferenceLatent,
)
from mlx_video.models.ltx_2.generate import (
    build_reference_context,
    create_position_grid,
    create_reference_position_grid,
    denoise_distilled,
    lora_reference_downscale_factor,
    read_lora_metadata,
)


def test_reference_positions_land_on_target_coordinates():
    # A 2x-downscaled reference token at latent (t, h, w) must sit at the same
    # pixel coordinate as the target token at latent (t, 2h, 2w).
    target = create_position_grid(1, 3, 4, 6)  # (1, 3, N, 2)
    ref = create_reference_position_grid(1, 3, 2, 3, downscale_factor=2)
    t = np.array(target)
    r = np.array(ref)
    assert r.shape == (1, 3, 3 * 2 * 3, 2)
    for ti in range(3):
        for hi in range(2):
            for wi in range(3):
                ref_idx = (ti * 2 + hi) * 3 + wi
                tgt_idx = (ti * 4 + 2 * hi) * 6 + 2 * wi
                # time axis untouched; spatial starts coincide
                np.testing.assert_allclose(r[0, 0, ref_idx], t[0, 0, tgt_idx])
                np.testing.assert_allclose(r[0, 1, ref_idx, 0], t[0, 1, tgt_idx, 0])
                np.testing.assert_allclose(r[0, 2, ref_idx, 0], t[0, 2, tgt_idx, 0])
                # a reference patch spans two target patches
                assert r[0, 1, ref_idx, 1] - r[0, 1, ref_idx, 0] == pytest.approx(64.0)
                assert r[0, 2, ref_idx, 1] - r[0, 2, ref_idx, 0] == pytest.approx(64.0)


def test_reference_positions_factor_one_is_plain_grid():
    a = create_position_grid(1, 2, 3, 3)
    b = create_reference_position_grid(1, 2, 3, 3, downscale_factor=1)
    np.testing.assert_array_equal(np.array(a), np.array(b))


def test_build_reference_context_shapes_and_mask():
    lat = mx.random.normal((1, 128, 2, 3, 4))
    ctx = build_reference_context(
        VideoConditionByReferenceLatent(lat, downscale_factor=2, strength=0.75),
        dtype=mx.bfloat16,
    )
    assert ctx.tokens.shape == (1, 24, 128)
    assert ctx.positions.shape == (1, 3, 24, 2)
    assert ctx.denoise_mask.shape == (1, 24)
    assert float(ctx.denoise_mask[0, 0]) == pytest.approx(0.25, abs=1e-2)
    assert ctx.num_tokens == 24


def test_condition_validation():
    with pytest.raises(ValueError):
        VideoConditionByReferenceLatent(mx.zeros((1, 128, 2, 3)))
    with pytest.raises(NotImplementedError):
        VideoConditionByReferenceLatent(mx.zeros((1, 128, 1, 1, 1)), temporal_scale_factor=2)
    with pytest.raises(ValueError):
        VideoConditionByReferenceLatent(mx.zeros((1, 128, 1, 1, 1)), strength=1.5)


class _RecordingTransformer:
    """Stands in for LTXModel: records what it was handed, returns zero velocity."""

    def __init__(self):
        self.calls = []

    def __call__(self, video=None, audio=None):
        self.calls.append(
            {
                "tokens": int(video.latent.shape[1]),
                "positions": int(video.positions.shape[2]),
                "timesteps": np.array(video.timesteps.astype(mx.float32)),
            }
        )
        return mx.zeros_like(video.latent), None


def test_denoise_distilled_appends_reference_and_slices_it_off():
    b, c, f, h, w = 1, 128, 2, 2, 2
    n = f * h * w
    latents = mx.random.normal((b, c, f, h, w))
    positions = create_position_grid(b, f, h, w)
    ref_lat = mx.random.normal((b, c, f, 1, 1))
    ctx = build_reference_context(
        VideoConditionByReferenceLatent(ref_lat, downscale_factor=2), dtype=mx.float32
    )
    model = _RecordingTransformer()
    text = mx.zeros((b, 4, 4096))

    out, audio = denoise_distilled(
        latents,
        positions,
        text,
        model,
        [1.0, 0.5, 0.0],
        verbose=False,
        reference=ctx,
    )
    assert audio is None
    assert out.shape == (b, c, f, h, w)  # reference never leaks into the output
    assert len(model.calls) == 2
    for call, sigma in zip(model.calls, [1.0, 0.5]):
        assert call["tokens"] == n + ctx.num_tokens
        assert call["positions"] == n + ctx.num_tokens
        ts = call["timesteps"][0]
        np.testing.assert_allclose(ts[:n], sigma)  # target noised at sigma
        np.testing.assert_allclose(ts[n:], 0.0)  # reference read clean
    # With zero velocity, x0 == latents, so the final Euler step returns the input.
    np.testing.assert_allclose(np.array(out), np.array(latents), atol=1e-5)


def test_denoise_distilled_without_reference_is_unchanged():
    b, c, f, h, w = 1, 128, 1, 2, 2
    latents = mx.random.normal((b, c, f, h, w))
    positions = create_position_grid(b, f, h, w)
    model = _RecordingTransformer()
    out, _ = denoise_distilled(
        latents, positions, mx.zeros((b, 4, 4096)), model, [1.0, 0.0], verbose=False
    )
    assert model.calls[0]["tokens"] == f * h * w
    assert out.shape == (b, c, f, h, w)


def _write_safetensors(path, metadata):
    header = {"__metadata__": metadata, "x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    raw = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)))
        fh.write(raw)
        fh.write(b"\x00\x00\x00\x00")


def test_lora_metadata_reader(tmp_path):
    p = tmp_path / "ic.safetensors"
    _write_safetensors(p, {"reference_downscale_factor": "2", "model_version": "2.5.0"})
    assert read_lora_metadata(p)["model_version"] == "2.5.0"
    assert lora_reference_downscale_factor(p) == 2
    q = tmp_path / "plain.safetensors"
    _write_safetensors(q, {})
    assert lora_reference_downscale_factor(q) == 1

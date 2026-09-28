"""Keyframe-index conditioning (an end frame) -- positions, contexts, wrapper.

The positions are checked against values computed by Lightricks' own
ltx-core (see fixtures/make_ltx_core_keyframe_positions.py); the rest pins
the in-context mechanics: what the transformer receives, and what it hands
back once the context tokens are cut away.
"""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_video.models.ltx_2.conditioning import (
    ReferenceContext,
    VideoConditionByKeyframeIndex,
    combine_contexts,
)
from mlx_video.models.ltx_2.generate import (
    InContextTransformer,
    build_keyframe_context,
    create_keyframe_position_grid,
    create_position_grid,
)
from mlx_video.models.ltx_2.transformer import Modality

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "ltx_core_keyframe_positions.json").read_text()
)


def _bf16(values) -> np.ndarray:
    """ltx-core's positions reach the model through a bfloat16 cast; ours are
    stored already cast, so the upstream values are cast the same way."""
    return np.array(mx.array(np.array(values, np.float32), dtype=mx.bfloat16).astype(mx.float32))


@pytest.mark.parametrize(
    "case", FIXTURE["cases"], ids=lambda c: f"frame{c['frame_idx']}@{c['fps']:g}fps"
)
def test_keyframe_positions_match_ltx_core(case):
    ours = create_keyframe_position_grid(
        1,
        case["latent_frames"],
        case["height"],
        case["width"],
        frame_idx=case["frame_idx"],
        num_pixel_frames=case["num_pixel_frames"],
        fps=case["fps"],
    )
    np.testing.assert_array_equal(np.array(ours), _bf16(case["positions"]))


def test_the_fixture_came_from_the_pinned_upstream():
    assert FIXTURE["upstream"].startswith("Lightricks/LTX-2@a95ab856")
    assert len(FIXTURE["cases"]) >= 6


def test_a_last_frame_sits_at_its_own_time_not_the_last_latent_slot():
    """25 frames at 24 fps: frame 24 starts at exactly 1 s and spans one
    pixel frame -- not the [17, 25) span of the last latent frame."""
    positions = np.array(create_keyframe_position_grid(1, 1, 1, 1, frame_idx=24, fps=24.0))
    start, end = positions[0, 0, 0]
    assert start == pytest.approx(1.0)
    assert end == pytest.approx(25 / 24, rel=1e-2)
    last_slot = np.array(create_position_grid(1, 4, 1, 1, fps=24.0))[0, 0, -1]
    assert not np.allclose([start, end], last_slot)


def test_frame_zero_keeps_the_causal_fix():
    """At frame 0 the grid is the target's own first frame, as upstream."""
    keyframe = np.array(create_keyframe_position_grid(1, 1, 2, 2, frame_idx=0))
    target = np.array(create_position_grid(1, 1, 2, 2))
    np.testing.assert_array_equal(keyframe[:, 0, :, 0], target[:, 0, :, 0])
    np.testing.assert_array_equal(keyframe[:, 1:], target[:, 1:])


def test_build_keyframe_context_flattens_tokens_and_sets_the_mask():
    latent = mx.arange(1 * 4 * 1 * 2 * 3, dtype=mx.float32).reshape(1, 4, 1, 2, 3)
    ctx = build_keyframe_context(
        VideoConditionByKeyframeIndex(latent=latent, frame_idx=24, strength=0.75), mx.float32
    )
    assert ctx.tokens.shape == (1, 6, 4)
    assert ctx.positions.shape == (1, 3, 6, 2)
    np.testing.assert_allclose(np.array(ctx.denoise_mask), np.full((1, 6), 0.25))
    # token n is latent position n across channels, in (f, h, w) order
    np.testing.assert_array_equal(
        np.array(ctx.tokens[0, 1]), np.array(latent[0, :, 0, 0, 1])
    )


def test_a_clean_keyframe_has_a_zero_mask():
    latent = mx.zeros((1, 4, 1, 1, 1))
    ctx = build_keyframe_context(VideoConditionByKeyframeIndex(latent=latent, frame_idx=8), mx.float32)
    np.testing.assert_array_equal(np.array(ctx.denoise_mask), [[0.0]])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"latent": mx.zeros((4, 1, 1, 1)), "frame_idx": 1}, "must be"),
        ({"latent": mx.zeros((1, 4, 1, 1, 1)), "frame_idx": -1}, "frame_idx"),
        ({"latent": mx.zeros((1, 4, 1, 1, 1)), "frame_idx": 1, "strength": 1.5}, "strength"),
        ({"latent": mx.zeros((1, 4, 1, 1, 1)), "frame_idx": 1, "num_pixel_frames": 0}, "pixel"),
    ],
)
def test_keyframe_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        VideoConditionByKeyframeIndex(**kwargs)


def _context(tokens: int, value: float, t0: float) -> ReferenceContext:
    return ReferenceContext(
        tokens=mx.full((1, tokens, 4), value),
        positions=mx.full((1, 3, tokens, 2), t0),
        denoise_mask=mx.full((1, tokens), value),
    )


def test_combine_contexts_concatenates_in_order():
    first, second = _context(2, 1.0, 0.0), _context(3, 2.0, 5.0)
    both = combine_contexts([first, second])
    assert both.num_tokens == 5
    np.testing.assert_array_equal(np.array(both.tokens[0, :, 0]), [1, 1, 2, 2, 2])
    np.testing.assert_array_equal(np.array(both.positions[0, 0, :, 0]), [0, 0, 5, 5, 5])
    np.testing.assert_array_equal(np.array(both.denoise_mask[0]), [1, 1, 2, 2, 2])


def test_combine_contexts_edge_cases():
    only = _context(1, 1.0, 0.0)
    assert combine_contexts([only]) is only
    assert combine_contexts([]) is None


class _Recorder:
    """A stand-in LTXModel: records what it was handed, returns ramps."""

    inner_dim = 8

    def __init__(self):
        self.calls = []

    def __call__(self, video=None, audio=None, **kwargs):
        self.calls.append((video, audio, kwargs))
        n = video.latent.shape[1]
        velocity = mx.broadcast_to(mx.arange(n, dtype=mx.float32)[None, :, None], (1, n, 4))
        return velocity, (None if audio is None else audio.latent)


def _video(tokens: int, sigma: float, rope=None) -> Modality:
    return Modality(
        latent=mx.zeros((1, tokens, 4)),
        timesteps=mx.full((1, tokens), sigma),
        positions=mx.zeros((1, 3, tokens, 2)),
        context=mx.zeros((1, 2, 8)),
        positional_embeddings=rope,
        sigma=mx.full((1,), sigma),
    )


def test_wrapper_appends_the_context_and_cuts_the_velocity():
    recorder = _Recorder()
    ctx = _context(2, 0.5, 9.0)  # mask 0.5 -> timestep sigma * 0.5
    wrapped = InContextTransformer(recorder, ctx)
    velocity, audio = wrapped(video=_video(3, 0.8), audio=None, stg_video_blocks=[1])

    sent, _, kwargs = recorder.calls[0]
    assert kwargs == {"stg_video_blocks": [1]}
    assert sent.latent.shape == (1, 5, 4)
    np.testing.assert_allclose(np.array(sent.timesteps[0]), [0.8, 0.8, 0.8, 0.4, 0.4])
    np.testing.assert_array_equal(np.array(sent.positions[0, 0, :, 0]), [0, 0, 0, 9, 9])
    assert velocity.shape == (1, 3, 4)
    np.testing.assert_array_equal(np.array(velocity[0, :, 0]), [0, 1, 2])
    assert audio is None


def test_wrapper_passes_audio_through_and_forwards_attributes():
    recorder = _Recorder()
    wrapped = InContextTransformer(recorder, _context(1, 0.0, 0.0))
    audio = Modality(
        latent=mx.ones((1, 2, 4)),
        timesteps=mx.zeros((1, 2)),
        positions=mx.zeros((1, 1, 2, 2)),
        context=mx.zeros((1, 2, 8)),
    )
    _, audio_out = wrapped(video=_video(2, 0.5), audio=audio)
    assert recorder.calls[0][1] is audio
    np.testing.assert_array_equal(np.array(audio_out), np.array(audio.latent))
    assert wrapped.inner_dim == 8


def test_wrapper_recomputes_precomputed_rope_once(monkeypatch):
    """Dev denoisers hand the model precomputed RoPE for the target alone;
    the wrapper must rebuild it for target + context, and only once."""
    import mlx_video.models.ltx_2.rope as rope_module

    seen = []

    def fake_freqs(positions, **kwargs):
        seen.append(positions.shape)
        return (mx.zeros((1,)), mx.zeros((1,)))

    monkeypatch.setattr(rope_module, "precompute_freqs_cis", fake_freqs)

    class _Model(_Recorder):
        positional_embedding_theta = 1.0
        positional_embedding_max_pos = (1, 1, 1)
        use_middle_indices_grid = True
        num_attention_heads = 1
        rope_type = "split"

        class config:
            double_precision_rope = False

    model = _Model()
    wrapped = InContextTransformer(model, _context(2, 0.0, 0.0))
    target_rope = (mx.ones((1,)), mx.ones((1,)))
    for _ in range(3):
        wrapped(video=_video(3, 0.5, rope=target_rope))
    assert seen == [(1, 3, 5, 2)]
    assert model.calls[0][0].positional_embeddings is not target_rope

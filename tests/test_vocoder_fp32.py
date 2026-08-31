"""The audio chain computes in fp32, whatever dtype the checkpoint ships.

Checkpoints are bf16; the reference implementation forces the entire
vocoder+BWE forward to fp32 because bf16 accumulation compounds through
~108 sequential convolutions and degrades spectral metrics by 40-90% --
audible as broadband hiss. The loaders hold the (small) models in fp32
and the BWE forward casts its input, so a bf16 mel cannot drag the chain
back down. The causal STFT also pads with ZEROS, like the reference --
replicating the first sample tinted the opening frames' mel."""

import math

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_video.models.ltx_2.audio_vae.vocoder import (
    MelSTFT,
    STFTFn,
    Vocoder,
    VocoderWithBWE,
    _load_vocoder_with_bwe,
)
from mlx_video.models.ltx_2.config import VocoderModelConfig


def _tiny_config() -> dict:
    """A checkpoint-shaped combined config small enough for a unit test:
    base upsamples x4 (hop 4 divides its output), BWE upsamples x12 =
    hop * (300/100), so residual and skip lengths meet exactly."""
    return {
        "vocoder": {
            "upsample_initial_channel": 8,
            "resblock": "AMP1",
            "upsample_rates": [2, 2],
            "upsample_kernel_sizes": [4, 4],
            "resblock_kernel_sizes": [3],
            "resblock_dilation_sizes": [[1]],
            "stereo": True,
            "use_tanh_at_final": False,
            "use_bias_at_final": False,
            "activation": "snakebeta",
        },
        "bwe": {
            "upsample_initial_channel": 8,
            "resblock": "AMP1",
            "upsample_rates": [3, 2, 2],
            "upsample_kernel_sizes": [5, 4, 4],
            "resblock_kernel_sizes": [3],
            "resblock_dilation_sizes": [[1]],
            "stereo": True,
            "use_tanh_at_final": False,
            "use_bias_at_final": False,
            "activation": "snakebeta",
            "apply_final_activation": False,
            "input_sampling_rate": 100,
            "output_sampling_rate": 300,
            "hop_length": 4,
            "n_fft": 8,
            "win_size": 8,
            "num_mels": 64,
        },
    }


def _reference_model(cfg: dict) -> VocoderWithBWE:
    """The same construction _load_vocoder_with_bwe performs -- its
    parameters, cast to bf16, stand in for a shipped checkpoint."""
    vocoder = Vocoder(VocoderModelConfig.from_dict(cfg["vocoder"]))
    bwe_config = VocoderModelConfig.from_dict(cfg["bwe"])
    bwe_config.apply_final_activation = False
    bwe = Vocoder(bwe_config)
    mel_stft = MelSTFT(
        filter_length=8, hop_length=4, win_length=8, n_mel_channels=64
    )
    return VocoderWithBWE(
        vocoder=vocoder,
        bwe_generator=bwe,
        mel_stft=mel_stft,
        input_sampling_rate=100,
        output_sampling_rate=300,
        hop_length=4,
    )


def test_combined_loader_holds_fp32_and_bf16_mel_cannot_downgrade_it():
    cfg = _tiny_config()
    weights = {
        key: value.astype(mx.bfloat16)
        for key, value in tree_flatten(_reference_model(cfg).parameters())
    }
    model = _load_vocoder_with_bwe(cfg, weights)

    dtypes = {value.dtype for _, value in tree_flatten(model.parameters())}
    assert dtypes == {mx.float32}

    mel = mx.random.normal((1, 2, 6, 64)).astype(mx.bfloat16)
    out = model(mel)
    assert out.dtype == mx.float32
    assert out.shape == (1, 2, 72)  # 6 frames x hop 4 x ratio 3


def test_stft_left_pad_is_zeros_not_a_replicated_first_sample():
    stft = STFTFn(filter_length=8, hop_length=2, win_length=8)
    stft.forward_basis = mx.ones_like(stft.forward_basis)
    magnitude, _ = stft(mx.ones((1, 16)))
    # Constant input is the discriminator: replicate padding made every
    # frame identical; zero padding leaves the opening frames lighter
    # until the window fills with real samples.
    assert float(magnitude[0, 0, 0]) < float(magnitude[0, -1, 0])
    assert abs(float(magnitude[0, -1, 0]) - 8 * math.sqrt(2)) < 1e-3

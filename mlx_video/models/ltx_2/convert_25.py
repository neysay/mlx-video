"""Convert LTX-2.5 split-component checkpoints to the MLX directory layout.

LTX-2.5 abandons the monolithic file: each component ships as its own
safetensors with its config embedded in the file's metadata header (plus
tokenizer assets packed as tensors in the text-encoder file). This converter
reads those metadata configs verbatim -- no shape sniffing -- and reuses the
proven key sanitizers from ``convert.py``.

Expected source directory (as downloaded from Lightricks/LTX-2.5):

    diffusion_models/ltx-2.5-22b-<variant>-transformer-bf16.safetensors
    text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
    vae/ltx-2.5-video-vae-conv-bf16.safetensors
    vae/ltx-2.5-audio-vae-bf16.safetensors
    latent_upscale_models/ltx-2.5-latent-spatial-upscaler-*.safetensors

Output layout matches convert.py, with two additions: ``text_encoder/``
(Gemma-4 weights + config) and ``tokenizer/`` are materialized from the
packed assets instead of symlinked from a HF snapshot.

Usage:
    python -m mlx_video.models.ltx_2.convert_25 \
        --source ~/.sayvoy-look/staging/ltx-2.5 --output LTX-2.5-distilled \
        --variant distilled
"""

import argparse
import json
import shutil
import struct
from pathlib import Path
from typing import Dict

import mlx.core as mx

from mlx_video.models.ltx_2.convert import (
    extract_text_projections,
    sanitize_audio_decoder,
    sanitize_audio_encoder,
    sanitize_transformer,
    sanitize_vae_decoder,
    sanitize_vae_encoder,
    sanitize_vocoder,
    save_config,
    save_sharded,
    save_single,
)

ARCH_VERSION = "2.5"


def read_metadata(path: Path) -> dict:
    """The safetensors __metadata__ block (string values, JSON payloads)."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header.get("__metadata__", {})


def find_component(source: Path, patterns: list) -> Path:
    for pattern in patterns:
        matches = sorted(source.rglob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"none of {patterns} found under {source}")


# ─── Transformer ─────────────────────────────────────────────────────────────


def transformer_config_from_metadata(meta_config: dict) -> dict:
    """Map the checkpoint's embedded transformer config onto the MLX config
    schema. Values come from the file; nothing here is inferred."""
    t = meta_config.get("transformer", meta_config)
    config = {
        "model_type": "ltx av model",
        "num_attention_heads": t["num_attention_heads"],
        "attention_head_dim": t["attention_head_dim"],
        "in_channels": t.get("in_channels", 128),
        "out_channels": t.get("out_channels", 128),
        "num_layers": t["num_layers"],
        "cross_attention_dim": t["cross_attention_dim"],
        "caption_channels": t.get("caption_channels", 3840),
        "audio_num_attention_heads": t.get("audio_num_attention_heads", 32),
        "audio_attention_head_dim": t.get("audio_attention_head_dim", 64),
        "audio_in_channels": t.get("audio_in_channels", 128),
        "audio_out_channels": t.get("audio_out_channels", 128),
        "audio_cross_attention_dim": t.get("audio_cross_attention_dim", 2048),
        "audio_caption_channels": t.get("audio_caption_channels", 3840),
        "positional_embedding_theta": t.get("positional_embedding_theta", 10000.0),
        "positional_embedding_max_pos": t.get(
            "positional_embedding_max_pos", [20, 2048, 2048]
        ),
        "audio_positional_embedding_max_pos": t.get(
            "audio_positional_embedding_max_pos", [20]
        ),
        "use_middle_indices_grid": t.get("use_middle_indices_grid", True),
        "rope_type": t.get("rope_type", "split"),
        "double_precision_rope": t.get("frequencies_precision") == "float64",
        "timestep_scale_multiplier": t.get("timestep_scale_multiplier", 1000),
        "av_ca_timestep_scale_multiplier": int(
            t.get("av_ca_timestep_scale_multiplier", 1000)
        ),
        "norm_eps": t.get("norm_eps", 1e-6),
        "has_prompt_adaln": bool(t.get("cross_attention_adaln", False)),
        "ff_bias": bool(t.get("ff_bias", True)),
        "audio_ff_bias": bool(t.get("audio_ff_bias", True)),
        "use_keyframes_abs_pos_embedding": bool(
            t.get("use_keyframes_abs_pos_embedding", False)
        ),
        "arch_version": ARCH_VERSION,
    }
    return config


# ─── Text encoder (Gemma 4) ──────────────────────────────────────────────────

_ENCODER_SKIP_PREFIXES = (
    "vision_model.",
    "multi_modal_projector.",
    "audio_projector.",
    "text_embedding_projection.",
)


def split_text_encoder(
    weights: Dict[str, mx.array],
) -> tuple[Dict[str, mx.array], Dict[str, bytes]]:
    """Gemma weights (model.*) and packed byte assets from the encoder file.

    Vision/audio projector towers are dropped: LTX-2.5 uses the text tower's
    hidden states only. The aggregate-embed projections are extracted
    separately into text_projections by the caller.
    """
    gemma: Dict[str, mx.array] = {}
    assets: Dict[str, bytes] = {}
    for key, value in weights.items():
        if key.startswith(_ENCODER_SKIP_PREFIXES):
            continue
        if key == "tokenizer_json" or key.startswith("hf_asset__"):
            import numpy as np

            name = "tokenizer.json" if key == "tokenizer_json" else key[
                len("hf_asset__"):
            ]
            assets[name] = np.array(value.astype(mx.uint8)).tobytes()
            continue
        if key.startswith("model."):
            if value.dtype == mx.float32:
                value = value.astype(mx.bfloat16)
            gemma[key] = value
    return gemma, assets


# ─── VAE key adaptation ──────────────────────────────────────────────────────


def _prefix_video_vae(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """The 2.5 conv VAE ships bare (Comfy-split) prefixes; re-namespace them
    so convert.py's sanitizers apply unchanged."""
    prefixed = {}
    for key, value in weights.items():
        if key.startswith(("decoder.", "encoder.", "per_channel_statistics.")):
            prefixed["vae." + key] = value
        else:
            prefixed[key] = value
    return prefixed


def vae_decoder_config_from_metadata(meta_config: dict) -> dict:
    vae = meta_config.get("vae", meta_config)
    return {
        "decoder_blocks": vae["decoder_blocks"],
        "decoder_base_channels": vae.get("decoder_base_channels", 128),
        "patch_size": vae.get("patch_size", 4),
        "timestep_conditioning": bool(vae.get("timestep_conditioning", False)),
        "spatial_padding_mode": vae.get("spatial_padding_mode", "zeros"),
        "causal_decoder": bool(vae.get("causal_decoder", False)),
        "latent_channels": vae.get("latent_channels", 128),
    }


def vae_encoder_config_from_metadata(meta_config: dict) -> dict:
    vae = meta_config.get("vae", meta_config)
    return {
        "convolution_dimensions": vae.get("dims", 3),
        "encoder_blocks": vae["encoder_blocks"],
        "encoder_spatial_padding_mode": vae.get("spatial_padding_mode", "zeros"),
        "in_channels": vae.get("in_channels", 3),
        "latent_log_var": vae.get("latent_log_var", "uniform"),
        "norm_layer": vae.get("norm_layer", "pixel_norm"),
        "out_channels": vae.get("latent_channels", 128),
        "patch_size": vae.get("patch_size", 4),
    }


# ─── Audio configs ───────────────────────────────────────────────────────────


def audio_configs_from_metadata(meta_config: dict) -> tuple[dict, dict, dict]:
    """(decoder, encoder, vocoder) configs.

    The audio family is unchanged from 2.3, so the proven constant configs
    from convert.py apply; the vocoder config is carried from the checkpoint
    verbatim because it already has the nested {vocoder, bwe} structure the
    BWE loader reads.
    """
    from mlx_video.models.ltx_2.convert import (
        infer_audio_encoder_config,
        infer_audio_vae_config,
    )

    vocoder = meta_config.get("vocoder", {})
    if not isinstance(vocoder, dict) or "vocoder" not in vocoder:
        vocoder = {"type": "bigvgan", "has_bwe_generator": True, **vocoder}
    return infer_audio_vae_config({}), infer_audio_encoder_config({}), vocoder


# ─── Main ────────────────────────────────────────────────────────────────────


def convert_25(source: str, output_path: Path, variant: str = "distilled") -> None:
    source_dir = Path(source)
    output_path.mkdir(parents=True, exist_ok=True)

    transformer_file = find_component(
        source_dir, [f"ltx-2.5-*-{variant}-transformer-*.safetensors"]
    )
    encoder_file = find_component(source_dir, ["gemma4-*-with-proj-*.safetensors"])
    video_vae_file = find_component(source_dir, ["ltx-2.5-video-vae-conv-*.safetensors"])
    audio_vae_file = find_component(source_dir, ["ltx-2.5-audio-vae-*.safetensors"])

    # 1. Transformer (+ connectors for text_projections)
    print(f"[1/5] Transformer: {transformer_file.name}")
    meta = read_metadata(transformer_file)
    transformer_meta_config = json.loads(meta["config"])
    weights = mx.load(str(transformer_file))
    transformer_weights = sanitize_transformer(weights)
    shards = save_sharded(transformer_weights, output_path / "transformer")
    save_config(
        transformer_config_from_metadata(transformer_meta_config),
        output_path / "transformer",
    )
    print(f"    {len(transformer_weights)} keys, {shards} shards")
    connector_weights = extract_text_projections(weights)
    del weights, transformer_weights

    # 2. Text encoder (Gemma 4) + tokenizer assets + projections
    print(f"[2/5] Text encoder: {encoder_file.name}")
    encoder_meta = read_metadata(encoder_file)
    gemma_config = json.loads(encoder_meta["gemma_config"])
    weights = mx.load(str(encoder_file))
    gemma_weights, assets = split_text_encoder(weights)
    save_sharded(gemma_weights, output_path / "text_encoder")
    save_config(gemma_config, output_path / "text_encoder")
    tokenizer_dir = output_path / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in assets.items():
        (tokenizer_dir / name).write_bytes(payload)
        print(f"    tokenizer asset: {name} ({len(payload)} bytes)")
    projection_weights = dict(connector_weights)
    projection_weights.update(
        extract_text_projections(weights)
    )  # aggregate embeds live here
    tp_dir = output_path / "text_projections"
    tp_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(tp_dir / "model.safetensors"), projection_weights)
    print(
        f"    {len(gemma_weights)} gemma keys, "
        f"{len(projection_weights)} projection keys"
    )
    del weights, gemma_weights

    # 3. Video VAE (conv variant)
    print(f"[3/5] Video VAE: {video_vae_file.name}")
    vae_meta = read_metadata(video_vae_file)
    vae_meta_config = json.loads(vae_meta["config"])
    weights = _prefix_video_vae(mx.load(str(video_vae_file)))
    decoder_weights = sanitize_vae_decoder(weights)
    save_single(decoder_weights, output_path / "vae" / "decoder")
    save_config(
        vae_decoder_config_from_metadata(vae_meta_config),
        output_path / "vae" / "decoder",
    )
    encoder_weights = sanitize_vae_encoder(weights)
    save_single(encoder_weights, output_path / "vae" / "encoder")
    save_config(
        vae_encoder_config_from_metadata(vae_meta_config),
        output_path / "vae" / "encoder",
    )
    print(f"    decoder {len(decoder_weights)} keys, encoder {len(encoder_weights)}")
    del weights

    # 4. Audio VAE + vocoder
    print(f"[4/5] Audio VAE: {audio_vae_file.name}")
    audio_meta = read_metadata(audio_vae_file)
    audio_meta_config = json.loads(audio_meta.get("config", "{}"))
    weights = mx.load(str(audio_vae_file))
    dec_cfg, enc_cfg, voc_cfg = audio_configs_from_metadata(audio_meta_config)
    audio_dec = sanitize_audio_decoder(weights)
    save_single(audio_dec, output_path / "audio_vae" / "decoder")
    save_config(dec_cfg, output_path / "audio_vae" / "decoder")
    audio_enc = sanitize_audio_encoder(weights)
    save_single(audio_enc, output_path / "audio_vae" / "encoder")
    save_config(enc_cfg, output_path / "audio_vae" / "encoder")
    vocoder_weights = sanitize_vocoder(weights)
    save_single(vocoder_weights, output_path / "vocoder")
    save_config(voc_cfg, output_path / "vocoder")
    print(
        f"    audio dec {len(audio_dec)}, enc {len(audio_enc)}, "
        f"vocoder {len(vocoder_weights)}"
    )
    del weights

    # 5. Upscalers
    print("[5/5] Upscalers")
    for upscaler in sorted(source_dir.rglob("*upscaler*.safetensors")):
        dest = output_path / upscaler.name
        if not dest.exists():
            shutil.copy2(str(upscaler), str(dest))
        print(f"    {upscaler.name}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert LTX-2.5 split-component checkpoints to MLX layout"
    )
    parser.add_argument("--source", required=True, help="Directory of 2.5 components")
    parser.add_argument("--output", required=True, help="Output model directory")
    parser.add_argument(
        "--variant", choices=["distilled", "dev"], default="distilled"
    )
    args = parser.parse_args()
    convert_25(args.source, Path(args.output), variant=args.variant)

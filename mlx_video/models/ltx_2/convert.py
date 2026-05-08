"""Convert LTX-2/2.3 safetensors to MLX directory layout.

Converts from the single-file format (e.g. Lightricks/LTX-2/ltx-2-19b-distilled.safetensors
or Lightricks/LTX-2.3/ltx-2.3-22b-distilled.safetensors) to the modular directory structure:

    output/
    ├── transformer/          # DiT transformer weights (sharded)
    │   ├── config.json
    │   ├── model-00001-of-N.safetensors
    │   └── model.safetensors.index.json
    ├── vae/
    │   ├── decoder/          # Video VAE decoder
    │   │   ├── config.json
    │   │   └── model.safetensors
    │   └── encoder/          # Video VAE encoder
    │       ├── config.json
    │       └── model.safetensors
    ├── audio_vae/
    │   ├── decoder/          # Audio VAE decoder
    │   │   ├── config.json
    │   │   └── model.safetensors
    │   └── encoder/          # Audio VAE encoder
    │       ├── config.json
    │       └── model.safetensors
    ├── vocoder/              # Audio vocoder
    │   ├── config.json
    │   └── model.safetensors
    └── text_projections/     # Text projection connectors
        └── model.safetensors

Usage:
    # From HF repo ID
    python -m mlx_video.models.ltx_2.convert --source Lightricks/LTX-2 --output LTX-2-distilled --variant distilled
    python -m mlx_video.models.ltx_2.convert --source Lightricks/LTX-2.3 --output LTX-2.3-distilled --variant distilled

    # From local folder containing the monolithic safetensors
    python -m mlx_video.models.ltx_2.convert --source ./Lightricks-LTX-2/ --output LTX-2-distilled --variant distilled

    # From a direct safetensors file path
    python -m mlx_video.models.ltx_2.convert --source ./ltx-2-19b-distilled.safetensors --output LTX-2-distilled --variant distilled
"""

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict

import mlx.core as mx

# ─── Key prefix routing ──────────────────────────────────────────────────────

TRANSFORMER_PREFIX = "model.diffusion_model."
VAE_DECODER_PREFIX = "vae.decoder."
VAE_ENCODER_PREFIX = "vae.encoder."
VAE_STATS_PREFIX = "vae.per_channel_statistics."
AUDIO_DECODER_PREFIX = "audio_vae.decoder."
AUDIO_ENCODER_PREFIX = "audio_vae.encoder."
AUDIO_STATS_PREFIX = "audio_vae.per_channel_statistics."
VOCODER_PREFIX = "vocoder."
TEXT_PROJ_PREFIX = "text_embedding_projection."
VIDEO_CONNECTOR_PREFIX = "model.diffusion_model.video_embeddings_connector."
AUDIO_CONNECTOR_PREFIX = "model.diffusion_model.audio_embeddings_connector."


# ─── Sanitization functions ──────────────────────────────────────────────────


def sanitize_transformer(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Sanitize transformer keys: strip prefix, rename layers, cast to bfloat16."""
    sanitized = {}
    for key, value in weights.items():
        if not key.startswith(TRANSFORMER_PREFIX):
            continue
        # Skip connector weights (they go to text_projections)
        if "audio_embeddings_connector" in key or "video_embeddings_connector" in key:
            continue

        new_key = key[len(TRANSFORMER_PREFIX) :]
        new_key = new_key.replace(".to_out.0.", ".to_out.")
        new_key = new_key.replace(".ff.net.0.proj.", ".ff.proj_in.")
        new_key = new_key.replace(".ff.net.2.", ".ff.proj_out.")
        new_key = new_key.replace(".audio_ff.net.0.proj.", ".audio_ff.proj_in.")
        new_key = new_key.replace(".audio_ff.net.2.", ".audio_ff.proj_out.")
        new_key = new_key.replace(".linear_1.", ".linear1.")
        new_key = new_key.replace(".linear_2.", ".linear2.")

        # Cast all weights to bfloat16 (matches MLX model loading behavior)
        if value.dtype != mx.bfloat16:
            value = value.astype(mx.bfloat16)

        sanitized[new_key] = value
    return sanitized


def sanitize_vae_decoder(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Sanitize VAE decoder keys: strip prefix, transpose Conv3d, wrap .conv."""
    sanitized = {}
    for key, value in weights.items():
        new_key = None

        if key.startswith(VAE_STATS_PREFIX):
            if key == "vae.per_channel_statistics.mean-of-means":
                new_key = "per_channel_statistics.mean"
            elif key == "vae.per_channel_statistics.std-of-means":
                new_key = "per_channel_statistics.std"
            else:
                continue
        elif key.startswith(VAE_DECODER_PREFIX):
            new_key = key[len(VAE_DECODER_PREFIX) :]
        else:
            continue

        # Conv3d weight transpose: PyTorch (O, I, D, H, W) -> MLX (O, D, H, W, I)
        if ".conv.weight" in key and value.ndim == 5:
            value = mx.transpose(value, (0, 2, 3, 4, 1))

        # Wrap .conv.weight -> .conv.conv.weight (CausalConv3d wrapper)
        if ".conv.weight" in new_key or ".conv.bias" in new_key:
            if ".conv.conv.weight" not in new_key and ".conv.conv.bias" not in new_key:
                new_key = new_key.replace(".conv.weight", ".conv.conv.weight")
                new_key = new_key.replace(".conv.bias", ".conv.conv.bias")

        sanitized[new_key] = value
    return sanitized


def sanitize_vae_encoder(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Sanitize VAE encoder keys: strip prefix, transpose Conv3d/Conv2d."""
    sanitized = {}
    for key, value in weights.items():
        new_key = None

        if "position_ids" in key:
            continue

        if key.startswith(VAE_STATS_PREFIX):
            if key == "vae.per_channel_statistics.mean-of-means":
                new_key = "per_channel_statistics.mean"
            elif key == "vae.per_channel_statistics.std-of-means":
                new_key = "per_channel_statistics.std"
            else:
                continue
            # Per-channel statistics must stay float32 for precision
            if value.dtype != mx.float32:
                value = value.astype(mx.float32)
        elif key.startswith(VAE_ENCODER_PREFIX):
            new_key = key[len(VAE_ENCODER_PREFIX) :]
        else:
            continue

        # Conv3d: PyTorch (O, I, D, H, W) -> MLX (O, D, H, W, I)
        if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 5:
            value = mx.transpose(value, (0, 2, 3, 4, 1))

        # Conv2d: PyTorch (O, I, H, W) -> MLX (O, H, W, I)
        if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 4:
            value = mx.transpose(value, (0, 2, 3, 1))

        sanitized[new_key] = value
    return sanitized


def sanitize_audio_decoder(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Sanitize audio VAE decoder keys: strip prefix, transpose Conv2d."""
    sanitized = {}
    for key, value in weights.items():
        new_key = None

        if key.startswith(AUDIO_DECODER_PREFIX):
            new_key = key[len(AUDIO_DECODER_PREFIX) :]
        elif key.startswith(AUDIO_STATS_PREFIX):
            if "mean-of-means" in key:
                new_key = "per_channel_statistics.mean_of_means"
            elif "std-of-means" in key:
                new_key = "per_channel_statistics.std_of_means"
            else:
                continue
        else:
            continue

        # Conv2d: PyTorch (O, I, H, W) -> MLX (O, H, W, I)
        if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 4:
            value = mx.transpose(value, (0, 2, 3, 1))

        sanitized[new_key] = value
    return sanitized


def sanitize_audio_encoder(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Sanitize audio VAE encoder keys: strip prefix, transpose Conv2d."""
    sanitized = {}
    for key, value in weights.items():
        new_key = None

        if key.startswith(AUDIO_ENCODER_PREFIX):
            new_key = key[len(AUDIO_ENCODER_PREFIX) :]
        elif key.startswith(AUDIO_STATS_PREFIX):
            if "mean-of-means" in key:
                new_key = "per_channel_statistics.mean_of_means"
            elif "std-of-means" in key:
                new_key = "per_channel_statistics.std_of_means"
            else:
                continue
        elif key == "latents_mean":
            new_key = "per_channel_statistics.mean_of_means"
        elif key == "latents_std":
            new_key = "per_channel_statistics.std_of_means"
        else:
            continue

        # Conv2d: PyTorch (O, I, H, W) -> MLX (O, H, W, I)
        if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 4:
            value = mx.transpose(value, (0, 2, 3, 1))

        sanitized[new_key] = value
    return sanitized


def sanitize_vocoder(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Sanitize vocoder keys: strip prefix, transpose Conv1d/ConvTranspose1d."""
    sanitized = {}
    for key, value in weights.items():
        if not key.startswith(VOCODER_PREFIX):
            continue

        new_key = key[len(VOCODER_PREFIX) :]

        # Handle Conv1d/ConvTranspose1d weight shape conversion
        if "weight" in new_key and value.ndim == 3:
            if "ups" in new_key:
                # ConvTranspose1d: PyTorch (in_ch, out_ch, kernel) -> MLX (out_ch, kernel, in_ch)
                value = mx.transpose(value, (1, 2, 0))
            else:
                # Conv1d: PyTorch (out_ch, in_ch, kernel) -> MLX (out_ch, kernel, in_ch)
                value = mx.transpose(value, (0, 2, 1))

        sanitized[new_key] = value
    return sanitized


def sanitize_connector_key(key: str) -> str:
    """Sanitize connector sub-key names."""
    key = key.replace(".ff.net.0.proj.", ".ff.proj_in.")
    key = key.replace(".ff.net.2.", ".ff.proj_out.")
    key = key.replace(".to_out.0.", ".to_out.")
    return key


def extract_text_projections(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Extract text projection weights (aggregate_embed + connectors).

    Handles both LTX-2 (aggregate_embed.weight) and LTX-2.3
    (video_aggregate_embed.*, audio_aggregate_embed.*) formats.
    """
    extracted = {}

    # aggregate_embed weights (text_embedding_projection.*)
    for key, value in weights.items():
        if key.startswith(TEXT_PROJ_PREFIX):
            new_key = key[len(TEXT_PROJ_PREFIX) :]
            extracted[new_key] = value

    # video_embeddings_connector
    for key, value in weights.items():
        if key.startswith(VIDEO_CONNECTOR_PREFIX):
            suffix = key[len(VIDEO_CONNECTOR_PREFIX) :]
            new_key = "video_embeddings_connector." + sanitize_connector_key(suffix)
            extracted[new_key] = value

    # audio_embeddings_connector
    for key, value in weights.items():
        if key.startswith(AUDIO_CONNECTOR_PREFIX):
            suffix = key[len(AUDIO_CONNECTOR_PREFIX) :]
            new_key = "audio_embeddings_connector." + sanitize_connector_key(suffix)
            extracted[new_key] = value

    return extracted


# ─── Saving utilities ─────────────────────────────────────────────────────────


def save_sharded(
    weights: Dict[str, mx.array],
    output_dir: Path,
    max_shard_size_bytes: int = 5 * 1024 * 1024 * 1024,  # 5GB per shard
):
    """Save weights as sharded safetensors with an index file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort keys for deterministic output
    sorted_keys = sorted(weights.keys())

    # Calculate total size
    total_size = sum(weights[k].nbytes for k in sorted_keys)

    # Determine sharding
    shards = []
    current_shard = {}
    current_size = 0

    for key in sorted_keys:
        tensor = weights[key]
        tensor_size = tensor.nbytes

        if current_size + tensor_size > max_shard_size_bytes and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0

        current_shard[key] = tensor
        current_size += tensor_size

    if current_shard:
        shards.append(current_shard)

    num_shards = len(shards)
    weight_map = {}

    for i, shard in enumerate(shards):
        if num_shards == 1:
            filename = "model.safetensors"
        else:
            filename = f"model-{i+1:05d}-of-{num_shards:05d}.safetensors"

        mx.save_safetensors(str(output_dir / filename), shard)

        for key in shard:
            weight_map[key] = filename

    # Write index
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    return num_shards


def save_single(weights: Dict[str, mx.array], output_dir: Path):
    """Save weights as a single safetensors file with an index."""
    output_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(output_dir / "model.safetensors"), weights)

    # Also write index for consistency
    total_size = sum(v.nbytes for v in weights.values())
    weight_map = {k: "model.safetensors" for k in sorted(weights.keys())}
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)


def save_config(config: dict, output_dir: Path):
    """Save config.json to a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
        f.write("\n")


# ─── Source resolution ─────────────────────────────────────────────────────────

# Matches monolithic model files: ltx-2-19b-distilled.safetensors, ltx-2.3-22b-dev.safetensors, etc.
MONOLITHIC_PATTERN = re.compile(
    r"^ltx-[\d.]+-\d+b-(?P<variant>distilled|dev)\.safetensors$"
)

# Matches upscaler files like ltx-2-spatial-upscaler-x2-1.0.safetensors,
# ltx-2.3-spatial-upscaler-x2-1.0.safetensors, etc.
UPSCALER_PATTERN = re.compile(
    r"^ltx-[\d.]+-(?:spatial|temporal)-upscaler-.+\.safetensors$"
)


def resolve_source(source: str, variant: str) -> Path:
    """Resolve source to a monolithic safetensors file path.

    Args:
        source: HF repo ID (e.g. "Lightricks/LTX-2"), local directory, or direct file path.
        variant: Model variant ("distilled" or "dev") to select the right file.

    Returns:
        Path to the monolithic safetensors file.
    """
    source_path = Path(source)

    # Direct file path
    if source_path.is_file():
        return source_path

    # Local directory — find the variant's safetensors file
    if source_path.is_dir():
        matches = []
        for f in sorted(source_path.glob("ltx-*b-*.safetensors")):
            m = MONOLITHIC_PATTERN.match(f.name)
            if m and m.group("variant") == variant:
                matches.append(f)

        if matches:
            return matches[0]

        # Broader fallback
        all_mono = sorted(source_path.glob("ltx-*.safetensors"))
        for f in all_mono:
            if variant in f.name and MONOLITHIC_PATTERN.match(f.name):
                return f

        raise FileNotFoundError(
            f"No monolithic *-{variant}.safetensors found in {source_path}. "
            f"Files found: {[f.name for f in all_mono]}"
        )

    # HF repo ID — download via huggingface_hub
    if "/" in source and not source_path.exists():
        from huggingface_hub import hf_hub_download, list_repo_files

        # Find the right file in the repo
        repo_files = list_repo_files(source)
        target = None
        for f in repo_files:
            m = MONOLITHIC_PATTERN.match(f)
            if m and m.group("variant") == variant:
                target = f
                break

        if not target:
            raise FileNotFoundError(
                f"No *-{variant}.safetensors found in {source}. "
                f"Available: {[f for f in repo_files if f.endswith('.safetensors')]}"
            )

        print(f"Downloading {target} from {source}...")
        local_path = hf_hub_download(repo_id=source, filename=target)
        return Path(local_path)

    raise FileNotFoundError(
        f"Source not found: {source}. Provide an HF repo ID, local directory, or file path."
    )


# ─── Config inference ─────────────────────────────────────────────────────────


def infer_transformer_config(weights: Dict[str, mx.array]) -> dict:
    """Infer transformer config from weight shapes."""
    # Count transformer layers
    max_layer = -1
    for key in weights:
        if "transformer_blocks." in key:
            parts = key.split(".")
            try:
                idx = parts.index("transformer_blocks") + 1
                if idx < len(parts) and parts[idx].isdigit():
                    max_layer = max(max_layer, int(parts[idx]))
            except ValueError:
                pass
    num_layers = max_layer + 1 if max_layer >= 0 else 48

    # Detect cross_attention_dim from attn2.to_k (cross-attention input dim)
    cross_attention_dim = 4096
    for key, value in weights.items():
        if "transformer_blocks.0.attn2.to_k.weight" in key:
            cross_attention_dim = value.shape[-1]
            break

    # Check for prompt_adaln_single (LTX-2.3 feature)
    has_prompt_adaln = any("prompt_adaln_single" in k for k in weights)

    config = {
        "attention_head_dim": 128,
        "attention_type": "default",
        "audio_attention_head_dim": 64,
        "audio_caption_channels": 3840,
        "audio_cross_attention_dim": 2048,
        "audio_in_channels": 128,
        "audio_num_attention_heads": 32,
        "audio_out_channels": 128,
        "audio_positional_embedding_max_pos": [20],
        "av_ca_timestep_scale_multiplier": 1000,
        "caption_channels": 3840,
        "cross_attention_dim": cross_attention_dim,
        "double_precision_rope": True,
        "in_channels": 128,
        "model_type": "ltx av model",
        "norm_eps": 1e-06,
        "num_attention_heads": 32,
        "num_layers": num_layers,
        "out_channels": 128,
        "positional_embedding_max_pos": [20, 2048, 2048],
        "positional_embedding_theta": 10000.0,
        "rope_type": "split",
        "timestep_scale_multiplier": 1000,
        "use_middle_indices_grid": True,
    }

    if has_prompt_adaln:
        config["has_prompt_adaln"] = True

    return config


def infer_vae_decoder_config(weights: Dict[str, mx.array], variant: str) -> dict:
    """Infer VAE decoder config from weights."""
    # Check for timestep conditioning keys
    has_timestep = any(
        "last_time_embedder" in k or "last_scale_shift_table" in k for k in weights
    )

    # Count channel multipliers from up_blocks
    max_block = -1
    for key in weights:
        if "up_blocks." in key:
            parts = key.split(".")
            try:
                idx = parts.index("up_blocks") + 1
                if idx < len(parts) and parts[idx].isdigit():
                    max_block = max(max_block, int(parts[idx]))
            except ValueError:
                pass

    # Default config
    config = {
        "ch": 128,
        "ch_mult": [1, 2, 4],
        "dropout": 0.0,
        "num_res_blocks": 2,
        "out_ch": 2,
        "resolution": 256,
        "timestep_conditioning": has_timestep,
        "z_channels": 8,
    }
    return config


def infer_vae_encoder_config(weights: Dict[str, mx.array]) -> dict:
    """Infer VAE encoder config from weight shapes.

    Block types and order are consistent across LTX versions.
    Multipliers and layer counts are inferred from the weights.
    """
    # Block layout: (index, type, stride_product)
    # res_x blocks use res_blocks keys; compress blocks use a single conv key
    block_layout = [
        (0, "res_x"),
        (1, "compress_space_res", 4),   # stride (1,2,2)
        (2, "res_x"),
        (3, "compress_time_res", 2),    # stride (2,1,1)
        (4, "res_x"),
        (5, "compress_all_res", 8),     # stride (2,2,2)
        (6, "res_x"),
        (7, "compress_all_res", 8),     # stride (2,2,2)
        (8, "res_x"),
    ]

    encoder_blocks = []
    for entry in block_layout:
        idx, block_type = entry[0], entry[1]
        if block_type == "res_x":
            num_layers = sum(
                1 for k in weights if k.startswith(f"down_blocks.{idx}.res_blocks.")
                and k.endswith(".conv1.conv.weight")
            )
            encoder_blocks.append(["res_x", {"num_layers": num_layers}])
        else:
            stride_product = entry[2]
            w = weights.get(f"down_blocks.{idx}.conv.conv.weight")
            in_channels = w.shape[-1]
            conv_out_channels = w.shape[0]
            out_channels = conv_out_channels * stride_product
            multiplier = out_channels // in_channels
            encoder_blocks.append([block_type, {"multiplier": multiplier}])

    return {
        "convolution_dimensions": 3,
        "encoder_blocks": encoder_blocks,
        "encoder_spatial_padding_mode": "zeros",
        "in_channels": 3,
        "latent_log_var": "uniform",
        "norm_layer": "pixel_norm",
        "out_channels": 128,
        "patch_size": 4,
    }


def infer_audio_vae_config(weights: Dict[str, mx.array]) -> dict:
    """Return audio VAE config."""
    return {
        "attn_resolutions": [],
        "attn_type": "vanilla",
        "causality_axis": "height",
        "ch": 128,
        "ch_mult": [1, 2, 4],
        "dropout": 0.0,
        "give_pre_end": False,
        "is_causal": True,
        "mel_bins": 64,
        "mel_hop_length": 160,
        "mid_block_add_attention": False,
        "norm_type": "pixel",
        "num_res_blocks": 2,
        "out_ch": 2,
        "resamp_with_conv": True,
        "resolution": 256,
        "sample_rate": 16000,
        "tanh_out": False,
        "z_channels": 8,
    }


def infer_audio_encoder_config(weights: Dict[str, mx.array]) -> dict:
    """Return audio encoder config (mirrors decoder but with encoder-specific fields)."""
    return {
        "attn_resolutions": [],
        "attn_type": "vanilla",
        "causality_axis": "height",
        "ch": 128,
        "ch_mult": [1, 2, 4],
        "dropout": 0.0,
        "in_channels": 2,
        "double_z": True,
        "is_causal": True,
        "mel_bins": 64,
        "mel_hop_length": 160,
        "mid_block_add_attention": False,
        "n_fft": 1024,
        "norm_type": "pixel",
        "num_res_blocks": 2,
        "resamp_with_conv": True,
        "resolution": 256,
        "sample_rate": 16000,
        "z_channels": 8,
    }


def infer_vocoder_config(weights: Dict[str, mx.array]) -> dict:
    """Infer vocoder config from weights."""
    # Check for bwe_generator (LTX-2.3 BigVGAN vocoder)
    has_bwe = any(k.startswith("bwe_generator") for k in weights)

    if has_bwe:
        return {
            "type": "bigvgan",
            "has_bwe_generator": True,
        }

    return {
        "output_sample_rate": 24000,
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "resblock_kernel_sizes": [3, 7, 11],
        "stereo": True,
        "upsample_initial_channel": 1024,
        "upsample_kernel_sizes": [16, 15, 8, 4, 4],
        "upsample_rates": [6, 5, 2, 2, 2],
    }


# ─── Main ─────────────────────────────────────────────────────────────────────


def convert(source: str, output_path: Path, variant: str = "distilled"):
    """Convert monolithic safetensors to modular directory layout.

    Args:
        source: HF repo ID (e.g. "Lightricks/LTX-2"), local directory, or file path.
        output_path: Output directory for the modular layout.
        variant: "distilled" or "dev".
    """
    source_path = resolve_source(source, variant)

    print(f"Loading monolithic weights from {source_path.name}...")
    all_weights = mx.load(str(source_path))
    total_keys = len(all_weights)
    print(f"  Loaded {total_keys} keys")

    # Route keys to components
    print("\nExtracting components...")

    # 1. Transformer
    print("  [1/7] Transformer...")
    transformer_weights = sanitize_transformer(all_weights)
    num_shards = save_sharded(transformer_weights, output_path / "transformer")
    config = infer_transformer_config(transformer_weights)
    save_config(config, output_path / "transformer")
    t_params = sum(v.size for v in transformer_weights.values())
    print(
        f"    {len(transformer_weights)} keys, {t_params:,} params, {num_shards} shards"
    )

    # 2. VAE Decoder
    print("  [2/7] VAE Decoder...")
    vae_decoder_weights = sanitize_vae_decoder(all_weights)
    save_single(vae_decoder_weights, output_path / "vae" / "decoder")
    config = infer_vae_decoder_config(vae_decoder_weights, variant)
    save_config(config, output_path / "vae" / "decoder")
    d_params = sum(v.size for v in vae_decoder_weights.values())
    print(f"    {len(vae_decoder_weights)} keys, {d_params:,} params")

    # 3. VAE Encoder
    print("  [3/7] VAE Encoder...")
    vae_encoder_weights = sanitize_vae_encoder(all_weights)
    save_single(vae_encoder_weights, output_path / "vae" / "encoder")
    config = infer_vae_encoder_config(vae_encoder_weights)
    save_config(config, output_path / "vae" / "encoder")
    e_params = sum(v.size for v in vae_encoder_weights.values())
    print(f"    {len(vae_encoder_weights)} keys, {e_params:,} params")

    # 4. Audio VAE Decoder
    print("  [4/7] Audio VAE Decoder...")
    audio_decoder_weights = sanitize_audio_decoder(all_weights)
    save_single(audio_decoder_weights, output_path / "audio_vae" / "decoder")
    config = infer_audio_vae_config(audio_decoder_weights)
    save_config(config, output_path / "audio_vae" / "decoder")
    a_params = sum(v.size for v in audio_decoder_weights.values())
    print(f"    {len(audio_decoder_weights)} keys, {a_params:,} params")

    # 5. Audio VAE Encoder
    print("  [5/7] Audio VAE Encoder...")
    audio_encoder_weights = sanitize_audio_encoder(all_weights)
    save_single(audio_encoder_weights, output_path / "audio_vae" / "encoder")
    config = infer_audio_encoder_config(audio_encoder_weights)
    save_config(config, output_path / "audio_vae" / "encoder")
    ae_params = sum(v.size for v in audio_encoder_weights.values())
    print(f"    {len(audio_encoder_weights)} keys, {ae_params:,} params")

    # 6. Vocoder
    print("  [6/7] Vocoder...")
    vocoder_weights = sanitize_vocoder(all_weights)
    save_single(vocoder_weights, output_path / "vocoder")
    config = infer_vocoder_config(vocoder_weights)
    save_config(config, output_path / "vocoder")
    v_params = sum(v.size for v in vocoder_weights.values())
    print(f"    {len(vocoder_weights)} keys, {v_params:,} params")

    # 7. Text Projections
    print("  [7/7] Text Projections...")
    text_proj_weights = extract_text_projections(all_weights)
    tp_dir = output_path / "text_projections"
    tp_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(tp_dir / "model.safetensors"), text_proj_weights)
    tp_params = sum(v.size for v in text_proj_weights.values())
    print(f"    {len(text_proj_weights)} keys, {tp_params:,} params")

    # Copy upscaler files
    print("\nCopying upscaler files...")
    source_dir = source_path.parent
    is_hf_repo = "/" in source and not Path(source).exists()
    upscaler_files = []

    if is_hf_repo:
        from huggingface_hub import list_repo_files

        upscaler_files = [
            f for f in list_repo_files(source) if UPSCALER_PATTERN.match(f)
        ]
    else:
        upscaler_files = [
            f.name
            for f in source_dir.iterdir()
            if f.is_file() and UPSCALER_PATTERN.match(f.name)
        ]

    if not upscaler_files:
        print("  No upscaler files found")

    for upscaler_file in sorted(upscaler_files):
        dest = output_path / upscaler_file
        if dest.exists():
            print(f"  {upscaler_file}: already exists, skipping")
            continue

        local_candidate = source_dir / upscaler_file
        if local_candidate.is_file():
            shutil.copy2(str(local_candidate), str(dest))
            print(f"  {upscaler_file}: copied")
        elif is_hf_repo:
            from huggingface_hub import hf_hub_download

            print(f"  {upscaler_file}: downloading from {source}...")
            downloaded = hf_hub_download(repo_id=source, filename=upscaler_file)
            shutil.copy2(downloaded, str(dest))
            print(f"  {upscaler_file}: done")
        else:
            print(f"  {upscaler_file}: not found, skipping")

    # Link text_encoder and tokenizer directories
    print("\nLinking text encoder & tokenizer...")
    for subdir in ["text_encoder", "tokenizer"]:
        dest = output_path / subdir
        if dest.exists():
            print(f"  {subdir}/: already exists, skipping")
            continue

        local_candidate = source_dir / subdir
        if local_candidate.is_dir():
            # Resolve through symlinks to get the real directory
            real_path = local_candidate.resolve()
            dest.symlink_to(real_path)
            print(f"  {subdir}/: symlinked to {real_path}")
        elif is_hf_repo:
            from huggingface_hub import list_repo_files, snapshot_download

            # Only download if the subdir exists in the repo
            repo_files = list_repo_files(source)
            if any(f.startswith(f"{subdir}/") for f in repo_files):
                print(f"  {subdir}/: downloading from {source}...")
                snapshot_download(
                    repo_id=source,
                    allow_patterns=f"{subdir}/*",
                    local_dir=str(output_path),
                )
                print(f"  {subdir}/: done")
            else:
                print(f"  {subdir}/: not in repo, skipping")
        else:
            print(f"  {subdir}/: not found in source, skipping")

    # Summary
    all_converted = (
        len(transformer_weights)
        + len(vae_decoder_weights)
        + len(vae_encoder_weights)
        + len(audio_decoder_weights)
        + len(audio_encoder_weights)
        + len(vocoder_weights)
        + len(text_proj_weights)
    )
    print(f"\nDone! Converted {all_converted}/{total_keys} keys")
    if all_converted < total_keys:
        known_prefixes = (
            TRANSFORMER_PREFIX,
            VAE_DECODER_PREFIX,
            VAE_ENCODER_PREFIX,
            VAE_STATS_PREFIX,
            AUDIO_DECODER_PREFIX,
            AUDIO_ENCODER_PREFIX,
            AUDIO_STATS_PREFIX,
            VOCODER_PREFIX,
            TEXT_PROJ_PREFIX,
            VIDEO_CONNECTOR_PREFIX,
            AUDIO_CONNECTOR_PREFIX,
        )
        skipped = [
            k for k in all_weights if not any(k.startswith(p) for p in known_prefixes)
        ]
        if skipped:
            print(f"  Skipped {len(skipped)} keys:")
            for k in sorted(skipped)[:20]:
                print(f"    {k}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert monolithic LTX-2/2.3 safetensors to modular MLX layout"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="HF repo ID (e.g. Lightricks/LTX-2, Lightricks/LTX-2.3), local directory, or direct safetensors file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for modular layout",
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["distilled", "dev"],
        default="distilled",
        help="Model variant (affects VAE decoder config and which file to download)",
    )
    args = parser.parse_args()

    convert(args.source, Path(args.output), variant=args.variant)

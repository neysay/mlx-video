"""Latent-based conditioning for I2V (Image-to-Video) generation.

This module provides conditioning that injects encoded image latents into
the video generation process at specific frame positions.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx


@dataclass
class VideoConditionByLatentIndex:
    """Condition video generation by injecting latents at a specific frame index.

    This replaces the latent at the specified frame index with the conditioned
    latent and controls how much denoising is applied via the strength parameter.

    Args:
        latent: Encoded image latent of shape (B, C, 1, H, W)
        frame_idx: Frame index to condition (0 = first frame)
        strength: Denoising strength (1.0 = full denoise, 0.0 = keep original)
    """

    latent: mx.array
    frame_idx: int = 0
    strength: float = 1.0

    def get_num_latent_frames(self) -> int:
        """Get number of latent frames in the conditioning."""
        return self.latent.shape[2]


@dataclass
class LatentState:
    """State for latent diffusion with conditioning support.

    Attributes:
        latent: Current noisy latent (B, C, F, H, W)
        clean_latent: Clean conditioning latent (B, C, F, H, W)
        denoise_mask: Per-frame denoising mask (B, 1, F, 1, 1) where
                      1.0 = full denoise, 0.0 = keep clean
    """

    latent: mx.array
    clean_latent: mx.array
    denoise_mask: mx.array

    def clone(self) -> "LatentState":
        """Create a copy of the state."""
        return LatentState(
            latent=self.latent,
            clean_latent=self.clean_latent,
            denoise_mask=self.denoise_mask,
        )


def create_initial_state(
    shape: Tuple[int, ...],
    seed: Optional[int] = None,
    noise_scale: float = 1.0,
) -> LatentState:
    """Create initial noisy latent state.

    Args:
        shape: Shape of latent (B, C, F, H, W)
        seed: Optional random seed
        noise_scale: Scale for initial noise (sigma)

    Returns:
        Initial LatentState with random noise
    """
    if seed is not None:
        mx.random.seed(seed)

    noise = mx.random.normal(shape)

    return LatentState(
        latent=noise * noise_scale,
        clean_latent=mx.zeros(shape),
        denoise_mask=mx.ones((shape[0], 1, shape[2], 1, 1)),  # Full denoise by default
    )


def apply_conditioning(
    state: LatentState,
    conditionings: List[VideoConditionByLatentIndex],
) -> LatentState:
    """Apply conditioning items to a latent state.

    Args:
        state: Current latent state
        conditionings: List of conditioning items to apply

    Returns:
        Updated LatentState with conditioning applied
    """
    state = state.clone()
    dtype = state.latent.dtype
    b, c, f, h, w = state.latent.shape

    for cond in conditionings:
        cond_latent = cond.latent
        frame_idx = cond.frame_idx
        strength = cond.strength

        # Validate shapes
        _, cond_c, cond_f, cond_h, cond_w = cond_latent.shape
        if (cond_c, cond_h, cond_w) != (c, h, w):
            raise ValueError(
                f"Conditioning latent spatial shape ({cond_c}, {cond_h}, {cond_w}) "
                f"does not match target shape ({c}, {h}, {w})"
            )

        if frame_idx >= f:
            raise ValueError(
                f"Frame index {frame_idx} is out of bounds for latent with {f} frames"
            )

        # Get the conditioning frames count
        num_cond_frames = cond_f
        end_idx = min(frame_idx + num_cond_frames, f)

        # Replace latent at conditioning position
        # state.latent[:, :, frame_idx:end_idx] = cond_latent[:, :, :end_idx - frame_idx]
        latent_list = []
        clean_list = []
        mask_list = []

        for i in range(f):
            if frame_idx <= i < end_idx:
                # Use conditioning latent
                cond_idx = i - frame_idx
                latent_list.append(cond_latent[:, :, cond_idx : cond_idx + 1])
                clean_list.append(cond_latent[:, :, cond_idx : cond_idx + 1])
                # Set mask: 1.0 - strength means less denoising for conditioned frames
                mask_list.append(mx.full((b, 1, 1, 1, 1), 1.0 - strength, dtype=dtype))
            else:
                # Keep original
                latent_list.append(state.latent[:, :, i : i + 1])
                clean_list.append(state.clean_latent[:, :, i : i + 1])
                mask_list.append(state.denoise_mask[:, :, i : i + 1])

        state.latent = mx.concatenate(latent_list, axis=2)
        state.clean_latent = mx.concatenate(clean_list, axis=2)
        state.denoise_mask = mx.concatenate(mask_list, axis=2)

    return state


def apply_denoise_mask(
    denoised: mx.array,
    clean: mx.array,
    denoise_mask: mx.array,
) -> mx.array:
    """Blend denoised output with clean state based on mask.

    Args:
        denoised: Denoised latent (B, C, F, H, W)
        clean: Clean conditioning latent (B, C, F, H, W)
        denoise_mask: Mask where 1.0 = use denoised, 0.0 = use clean

    Returns:
        Blended latent
    """
    one = mx.array(1.0, dtype=denoised.dtype)
    return denoised * denoise_mask + clean * (one - denoise_mask)


def add_noise_with_state(
    state: LatentState,
    noise_scale: float,
) -> LatentState:
    """Add noise to state while respecting conditioning.

    For conditioned frames (mask < 1.0), adds noise proportionally
    to allow some refinement while preserving the conditioning.

    Args:
        state: Current latent state
        noise_scale: Scale for noise (sigma)

    Returns:
        Updated state with noise added
    """
    state = state.clone()

    # Generate noise
    noise = mx.random.normal(state.latent.shape)

    # For fully conditioned frames (mask=0), we want to add minimal noise
    # For unconditioned frames (mask=1), we want full noise
    # noisy = noise * sigma + latent * (1 - sigma)
    # But we scale sigma by the mask for conditioned regions

    effective_scale = noise_scale * state.denoise_mask
    one = mx.array(1.0, dtype=state.latent.dtype)
    state.latent = noise * effective_scale + state.latent * (one - effective_scale)

    return state


# ---------------------------------------------------------------------------
# In-context reference conditioning (IC-LoRA)
# ---------------------------------------------------------------------------


@dataclass
class VideoConditionByReferenceLatent:
    """Condition generation on a reference video read *in context* (IC-LoRA).

    The reference latent is flattened to tokens and appended after the
    target's tokens on the sequence axis for every transformer call; it is
    never denoised and is sliced off again before the x0 step. This mirrors
    ``ltx_core.conditioning.types.reference_video_cond`` in Lightricks'
    LTX-2 reference implementation.

    Args:
        latent: Encoded reference video, shape (B, C, F', H', W'). Spatially it
            is the target's latent grid divided by ``downscale_factor`` (the
            adapter's ``reference_downscale_factor`` metadata).
        downscale_factor: How much smaller the reference is than the target,
            per axis. Its RoPE positions are multiplied by this so a reference
            token sits at the same pixel coordinate as the target region it
            describes.
        temporal_scale_factor: Temporal subsampling of the reference. Only 1 is
            supported for now.
        strength: 1.0 keeps the reference tokens clean (timestep 0). Lower
            values noise them like the target (timestep ``sigma * (1 - strength)``).
    """

    latent: mx.array
    downscale_factor: int = 1
    temporal_scale_factor: int = 1
    strength: float = 1.0

    def __post_init__(self) -> None:
        if self.latent.ndim != 5:
            raise ValueError(
                f"reference latent must be (B, C, F, H, W), got shape {self.latent.shape}"
            )
        if self.downscale_factor < 1:
            raise ValueError("downscale_factor must be >= 1")
        if self.temporal_scale_factor != 1:
            raise NotImplementedError(
                "temporal_scale_factor != 1 is not supported by the in-context path yet"
            )
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("strength must be in [0, 1]")


@dataclass
class ReferenceContext:
    """A reference video prepared for the transformer's sequence axis.

    Built once per denoise call (see ``generate.build_reference_context``) and
    concatenated after the target tokens on every step.

    Attributes:
        tokens: Reference tokens, (B, M, C) -- the flattened clean latent.
        positions: RoPE positions for those tokens, (B, 3, M, 2), in the
            target's pixel space (already scaled by the downscale factor).
        denoise_mask: Per-token mask (B, M), ``1 - strength``; multiplied by
            the step's sigma to give each reference token its timestep.
    """

    tokens: mx.array
    positions: mx.array
    denoise_mask: mx.array

    @property
    def num_tokens(self) -> int:
        return int(self.tokens.shape[1])

"""Regenerate ltx_core_keyframe_positions.json from Lightricks' own code.

The keyframe-conditioning parity test (tests/test_keyframe_conditioning.py)
compares our RoPE positions for a keyframe against what ltx-core computes.
This script is how those expected values were made: it runs upstream's
UNMODIFIED ``VideoConditionByKeyframeIndex.apply_to`` (and the patchifier
and pixel-coordinate code it calls) under torch.

It needs torch and a checkout of ltx-core's sources (pinned below); this
repo's environment has neither, so run it elsewhere, e.g.:

    PYTHONPATH=/path/to/LTX-2/packages/ltx-core/src \\
        python tests/fixtures/make_ltx_core_keyframe_positions.py

Pinned upstream: Lightricks/LTX-2 @ a95ab856bf29407b6b066ede0abe1846050db56c
(2026-08-26), packages/ltx-core/src/ltx_core.
"""

import json
from pathlib import Path

import torch
from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex
from ltx_core.tools import VideoLatentTools
from ltx_core.types import VideoLatentShape

UPSTREAM = "Lightricks/LTX-2@a95ab856bf29407b6b066ede0abe1846050db56c"

#: (target latent frames, keyframe pixel frame, fps, latent h, latent w, keyframe latent frames, pixel frames)
CASES = [
    (4, 0, 24.0, 2, 3, 1, 1),  # first frame: causal fix applies
    (4, 24, 24.0, 2, 3, 1, 1),  # last pixel frame of a 25-frame clip
    (7, 48, 24.0, 2, 3, 1, 1),  # last of 49
    (13, 96, 25.0, 3, 2, 1, 1),  # last of 97, at 25 fps
    (4, 13, 24.0, 2, 2, 1, 1),  # an interior frame off the latent grid
    (4, 9, 24.0, 2, 2, 2, 9),  # a 2-latent-frame (9 pixel frame) clip keyframe
]


def main() -> None:
    cases = []
    for frames, frame_idx, fps, h, w, key_frames, pixel_frames in CASES:
        target = VideoLatentShape(batch=1, channels=128, frames=frames, height=h, width=w)
        tools = VideoLatentTools(
            patchifier=VideoLatentPatchifier(patch_size=1), target_shape=target, fps=fps
        )
        state = tools.create_initial_state(device="cpu", dtype=torch.float32)
        keyframe = torch.zeros(1, 128, key_frames, h, w)
        out = VideoConditionByKeyframeIndex(
            keyframes=keyframe, frame_idx=frame_idx, strength=1.0, num_pixel_frames=pixel_frames
        ).apply_to(state, tools)
        appended = out.positions[:, :, state.positions.shape[2] :, :]
        cases.append(
            {
                "latent_frames": key_frames,
                "height": h,
                "width": w,
                "frame_idx": frame_idx,
                "fps": fps,
                "num_pixel_frames": pixel_frames,
                "positions": appended.to(torch.float32).tolist(),
                "denoise_mask": out.denoise_mask[:, state.positions.shape[2] :, 0].tolist(),
            }
        )
    path = Path(__file__).with_name("ltx_core_keyframe_positions.json")
    path.write_text(json.dumps({"upstream": UPSTREAM, "cases": cases}, indent=1))
    print(f"wrote {path} ({len(cases)} cases)")


if __name__ == "__main__":
    main()

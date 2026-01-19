#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2024-2026 cubicibo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

######################
## Blu-ray PIP with alpha overlay script
##
##    Helper class to produce a secondary video that can be overlaid smoothly on a main program
##     using the Blu-ray Picture-In-Picture system, with proper disc authoring software.
##
##    AUTHORING STEPS:
##     - Set secondary video luma key feature at authoring, with the right luma threshold (see readme)
##     - Ensure secondary video is not rescaled (pip scale must be set to 1x)
##     e.g. https://blu-disc.net/download/manuals/user_guide_standard_mx.pdf - pp391 "Scaling: No"
##     DO NOT "full screen scaling" as the video is resampled and this could lead to bad luma keying.
##
######################

from numba import njit, prange
from pathlib import Path
from typing import Union
import numpy as np
import os

import vapoursynth as vs
core = vs.core

@njit(fastmath=True, parallel=True, forceinline=True)
def _vect_mod16(npf):
    max_val = np.iinfo(npf.dtype).max
    h, w = npf.shape
    w = int(np.ceil(w/16))
    for y in prange(int(np.ceil(h/16))):
        yp = y << 4
        for x in prange(w):
            xp = x << 4
            if np.any(npf[yp:yp+16, xp:xp+16]):
                npf[yp:yp+16, xp:xp+16] = max_val
    return npf

class PIPelette:
    _warned_not_444p = 0
    def __init__(self,
        min_luma: int = 16,
        pad_mod16: bool = True,
        matrix: Union[int, str, 'MatrixCoefficients', None] = None,
        append_clear_frame: Union[bool, int] = True,
        *,
        default_pip_clip_luma: int = 0,
    ) -> None:
        """
        Sets up the PIPelette internals.

        Args:
            min_luma: min_luma in the PiP overlay that isn't masked. Should always be 16 (default of limited range)
            pad_mod16: pad overlay borders to the macroblock grid to avoid smoothing
            matrix: matrix for RGB overlay, if None: guessed from the clip height.
            append_clear_frame: Append one or more fully transparent frame at the end of the overlay clip.
            default_pip_clip_luma: luma value to use for the darker than black area, should be 0.
        """
        if not isinstance(append_clear_frame, (bool, int)):
            raise TypeError(f"appeand_clear_frame not bool, got '{append_clear_frame}'")
        if not isinstance(pad_mod16, (bool, int)):
            raise TypeError(f"pad_mod16 not bool, got '{pad_mod16}'")
        self.append_clear_frame = append_clear_frame
        self.pad_mod16 = pad_mod16
        self.min_luma = min_luma
        self.matrix = matrix
        if default_pip_clip_luma not in range(0, 16):
            raise ValueError(f"Incorrect default luma value for PIP clip, got '{default_pip_clip_luma}'.")
        self.default_pip_clip_luma = int(default_pip_clip_luma)

    @property
    def min_luma(self):
        return self._min_luma

    @min_luma.setter
    def min_luma(self, min_luma: int) -> None:
        if not isinstance(min_luma, int):
            raise TypeError("Incorrect type for min_luma.")
        if min_luma not in range(1, 235):
            raise ValueError("min_luma of PiP clip outside of range [1; 235[.")
        self._min_luma = min_luma

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, matrix: Union[int, str, 'MatrixCoefficients', None]) -> None:
        if isinstance(matrix, str):
            if '709' in matrix:
                matrix = vs.MATRIX_BT709
            elif '170' in matrix:
                matrix = vs.MATRIX_ST170_M
            elif '240' in matrix:
                matrix = vs.MATRIX_ST240_M
            elif '470' in matrix:
                matrix = vs.MATRIX_BT470_BG
            else:
                raise ValueError("Unrecognised string for matrix, got '{matrix}'.")
        else:
            self._matrix = matrix

    @staticmethod
    def _mod16ify_bmask(n, f) -> vs.VideoFrame:
        vsf = f.copy()
        np.copyto(np.asarray(vsf[0]), _vect_mod16(np.asarray(vsf[0])))
        return vsf

    def _check_and_setup(self,
        clip: vs.VideoNode,
        overlay: vs.VideoNode,
        mask: vs.VideoNode
    ) -> tuple[vs.VideoNode, vs.VideoNode, vs.VideoNode]:
        if clip.format.color_family != vs.YUV:
            raise TypeError(f"clip isn't YUV, got '{clip.format.color_family}'.")

        base_props = clip.get_frame(0).props
        overlay_props = overlay.get_frame(0).props

        if base_props['_ColorRange'] != vs.RANGE_LIMITED:
            raise ValueError("clip isn't limited range.")

        if overlay.format.color_family != vs.YUV:
            if overlay.format.color_family != vs.RGB:
                raise ValueError("overlay clip is neither YUV or RGB.")
            matrix = self.matrix
            if matrix is None:
                matrix = { # BD supports 480, 576, 720 or 1080. UHD BD does NOT support PIP.
                    480: vs.MATRIX_ST170_M, 576: vs.MATRIX_BT470_BG, 720: vs.MATRIX_BT709, 1080: vs.MATRIX_BT709,
                }.get(clip.height)
            if matrix is None:
                raise ValueError("No matrix specified and unable to deduce a valid one.")
            # Set-up overlay to match input
            overlay = core.resize.Bilinear(overlay, format=clip.format, range=vs.RANGE_LIMITED, matrix=matrix)
        else:
            if overlay_props['_Matrix'] != base_props['_Matrix']:
                raise ValueError(f"matrix mismath between base ('{base_matrix}') and overlay ('{overlay_matrix}').")
            if overlay_props['_ColorRange'] != vs.RANGE_LIMITED:
                raise ValueError("overlay isn't limited range.")

        assert clip.format.bits_per_sample == overlay.format.bits_per_sample == mask.format.bits_per_sample, "bitdepth must match for all assets."
        assert mask.format.color_family == vs.GRAY and 1 == mask.format.num_planes, "Only grayscale for the mask."
        assert clip.format.bits_per_sample == 8, "Only 8-bit depth for BDAV."
        assert clip.width == mask.width and clip.height == mask.height, "clip and mask shapes must match."

        if clip.format.id not in (vs.YUV444P8,) and __class__._warned_not_444p == 0:
            import warnings
            warnings.warn(f"clip isn't YUV444P (got '{clip.format.name}'), chroma may be processed incorrectly.")
            __class__._warned_not_444p += 1

        return clip, overlay, mask

    def apply(self,
            clip: vs.VideoNode,
            overlay: vs.VideoNode,
            mask: vs.VideoNode | None = None,
            merge_overlay: bool = True,
            length: int | None = None,
        ) -> vs.VideoNode:
        """
        Generate the smooth overlay clip to use as secondary video.

        Args:
            clip: primary feature
            overlay: clip to display smoothly on top of primary
            mask: alpha mask for the overlay. If None, the overlay must have a _Alpha prop.
            merge_overlay: If True (default) the secondary video contains the overlay. If False, the secondary video
             is a mask sized to hide the overlay from the primary.
             E.g. "False" could mean primary contains the hardsub, and the secondary video can be used to hide it.
            length: If set: output duration in frames, else inherit from clip or overlay, whichever is shorter.
        Returns:
            overlay clip to use as secondary video
        """
        if mask is None:
            mask = core.std.PropToClip(overlay, "_Alpha")

        clip, overlay, mask = self._check_and_setup(clip, overlay, mask)

        if merge_overlay:
            combined = core.std.MaskedMerge(clip, overlay, mask)
        else:
            combined = clip
        combined = core.std.Limiter(combined, min=self.min_luma, planes=[0])

        binmask = core.std.Binarize(mask, threshold=1)
        if self.pad_mod16:
            binmask = core.std.ModifyFrame(binmask, binmask, PIPelette._mod16ify_bmask)

        # The overlay decoder WILL underflow when it reaches the end of the stream.
        # This means the frame will NOT be erased from the display. This is a critical issue if the PIP
        # is shorter than the main program. Appending a transparent frame ensures the undisplay prior to the underflow
        # Also enforce that pip length does not exceed base clip
        if length is None:
            length = len(clip)
        length = min(len(clip), int(self.append_clear_frame) + min(len(overlay), len(mask), length))

        pip_clip = core.std.BlankClip(combined, length=length, color=[self.default_pip_clip_luma, 128, 128])

        pip_clip = core.std.MaskedMerge(pip_clip, combined, binmask)
        pip_clip = core.std.SetFrameProp(pip_clip, prop="_ColorRange", intval=int(vs.RANGE_LIMITED))
        return pip_clip
####

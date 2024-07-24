#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2024 cubicibo

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
##    This script produces a secondary video that can be overlaid smoothly on a main program
##     using the Blu-ray Picture-In-Picture system, with proper disc authoring software.
##
##    AUTHORING STEPS:
##     - Set secondary video luma key value to Y = 0 (or LUMA_KEY_LOWER_BOUND - 1)
##     - Ensure secondary video is not rescaled (pip scale must be set to 1x)
##     e.g. https://blu-disc.net/download/manuals/user_guide_standard_mx.pdf - pp391 "Scaling: No"
##     DO NOT "full screen scaling" as the video is resampled and this could lead to bad luma keying.
##
##
## Key variables: 
##     - clip            main program
##     - overlay         content to overlay on the main program): some clip with an alpha mask
##                        like an After Effect export, or ASS script obtained by MaskSub, see example below
##     - target_matrix   target matrix of main program, to convert overlay to clip colour space.
##                        None = default value, inferred from clip height.
##
## Misc (should not be modified unless...):
##     - mask            the overlay alpha mask that defines the blended region with clip. mask is fetched
##                        from overlay automatically, but a separate clip could also be provided too.
##
######################

from numba import njit, prange
from pathlib import Path
from functools import partial
import numpy as np
import os

import vapoursynth as vs
core = vs.core

#################################
###################### PARAMETERS

# load ~/test folder and plugins
upwd = Path(os.path.expanduser('~')).joinpath('test')
upwd_plugins = Path(upwd).joinpath('plugins')

core.std.LoadPlugin(upwd_plugins.joinpath('ffms2', 'src', 'core', '.libs', 'libffms2.so.4.0.0'))
core.std.LoadPlugin(upwd_plugins.joinpath('subtext', 'build', 'libsubtext.so'))

# MAIN PROGRAM
clip = core.ffms2.Source(Path(upwd).joinpath("nc2.mp4"))
clip = core.resize.Point(clip, format=vs.YUV444P8) #Process everything in 444P

target_matrix = None # None => guess from clip height

# OVERLAY PROGRAM with mask (e.g. ASS script)
script = Path(upwd).joinpath("nc2.ass")
additional_fontdir = "" #"/mnt/c/Windows/Fonts/" # EMPTY STRING if unneeded

overlay = core.sub.TextFile(clip, file=script, fontdir=additional_fontdir, blend=False)
mask = core.std.PropToClip(clip=overlay, prop='_Alpha')

###################### PARAMETERS
#################################

PAD_MOD16 = True          # PIP mask is mod16 aligned, highly recommended
LUMA_KEY_LOWER_BOUND = 1  # Min LUMA (Y) value of the PIP clip. CANNOT be zero.
CLIP_LUMA = True          # True: Y=0 to LUMA_KEY_LOWER_BOUND-1 are clipped to LUMA_KEY_LOWER_BOUND.
                          #  False: all luma is remapped in LUMA_KEY_LOWER_BOUND;255.

# BD supports 480, 576, 720 or 1080. UHD BD does NOT support PIP.
lut_matrix_height = {
    480: vs.MATRIX_ST170_M,
    576: vs.MATRIX_BT470_BG,
    720: vs.MATRIX_BT709,
    1080: vs.MATRIX_BT709,
}

if target_matrix is None:
    target_matrix = lut_matrix_height.get(clip.height, None)
assert target_matrix is not None

###################### FUNCTIONS, no need to read further

@njit(fastmath=True, parallel=True, forceinline=True)
def vect_mod16(npf):
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

def mod16ify_bmask(n, f) -> vs.VideoFrame:
    vsf = f.copy()
    np.copyto(np.asarray(vsf[0]), vect_mod16(np.asarray(vsf[0])))
    return vsf

def f_clip_luma(n, f, *, min_luma: int) -> vs.VideoFrame:
    vsf = f.copy()
    np.copyto(np.asarray(vsf[0]), np.clip(np.asarray(vsf[0]), min_luma, None))
    return vsf

def set_pip_visible_range(
    clip: vs.VideoNode, *, min_luma_out: int = LUMA_KEY_LOWER_BOUND, clip_luma: bool = CLIP_LUMA
) -> vs.VideoNode:
    assert min_luma_out > 0
    if clip_luma:
        clip = core.std.ModifyFrame(clip, clip, partial(f_clip_luma, min_luma=min_luma_out))
    else:
        clip = core.std.Levels(clip, min_in=0, max_in=255, min_out=min_luma_out, max_out=255, planes=0)
    return clip

def PIPelette(
    clip: vs.VideoNode, overlay: vs.VideoNode, mask: vs.VideoNode, *, pad_mod16: bool = PAD_MOD16, append_frame: bool = True,
) -> vs.VideoNode:
    """
    clip:    main program, YUV limited range
    overlay: content to overlay on the main program with a second video stream, YUV limited range.
    mask:    blend mask to use if the overlay were to be burned in the main program, grayscale.

    append_frame: flag to append a single, fully transparent, frame to prevent mismatched clip-overlay issues.

    :return: secondary video to overlay via BD PIP framework.
    """
    assert clip.format.color_family == vs.YUV == overlay.format.color_family, "Only YUV clips."
    assert clip.format.bits_per_sample == overlay.format.bits_per_sample == mask.format.bits_per_sample, "depth mismatch."
    assert mask.format.color_family == vs.GRAY and 1 == mask.format.num_planes, "Only 2D grayscale mask."
    assert clip.format.bits_per_sample == 8, "Only 8-bit depth for BDAV."

    burned_clip = core.std.MaskedMerge(clip, overlay, mask)
    burned_clip = set_pip_visible_range(burned_clip)

    binmask = core.std.Binarize(mask, 1)
    if pad_mod16:
        binmask = core.std.ModifyFrame(binmask, binmask, mod16ify_bmask)

    # The secondary video (=PIP video) decoder WILL underflow when it reaches the end of the stream.
    # This means the frame will NOT be erased from the display. This is a critical issue if the PIP
    # is shorter than the main program. Appending a transparent frame ensures undisplay before underflow.
    blank = core.std.BlankClip(burned_clip, color=[0, 128, 128])

    pip_clip = core.std.MaskedMerge(blank, burned_clip, binmask)
    pip_clip = core.std.SetFrameProp(pip_clip, prop="_ColorRange", intval=int(vs.RANGE_LIMITED))

    if append_frame:
        pip_clip = pip_clip + core.std.BlankClip(pip_clip, length=1, color=[0, 128, 128])
    return pip_clip

def run(clip, overlay, mask, target_matrix):
    overlay = core.resize.Point(overlay, format=clip.format, matrix=target_matrix)
    clip = PIPelette(clip, overlay, mask)
    clip.set_output()
####

run(clip, overlay, mask, target_matrix)
#

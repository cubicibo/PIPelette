# PIPelette
VapourSynth script to prepare Picture-in-Picture assets for BDAV.

## Status
Proof of concept. You must modify the script inputs manually, and update the VS plugins path to your needs.

The current script produces a secondary video from an ASS script that may include complex typesetting effects that cannot be encoded as PGS.

## Principle
PIPelette is a VapourSynth script that, given:
- a main program (primary video)
- a video track to blend smoothly with the main program.
- an alpha mask tied to the overlay.

will prepare a secondary video that can be encoded with your favorite AVC encoder (x264) and imported in your authoring software as a secondary video track. In the professional authoring software, you will need to:
- Enable the picture-in-picture luma-key feature, with a cut-off luma at Y=0.
- Set the PIP scale to be 1x (primary video = PIP video scale).

## Requirements
- VapourSynth
- numpy
- numba

# PIPelette
VapourSynth class to prepare Picture-in-Picture assets for BDAV.

## Principle
PIPelette is a VapourSynth class that, given:
- a main program (primary video)
- an overlay track to display smoothly on top of the main program.
- an alpha mask tied to the overlay.

will prepare a secondary video to import in your authoring software as a secondary video track. In the authoring software, you will need to:
- Enable the picture-in-picture luma-key feature, with a cut-off luma at Y < 16 (recommended: Y=8).
- Set the PIP scale to be 1x (primary video = PIP video scale).

On playback of the main program on a commercial Blu-ray Disc player, the secondary video track can be enabled (forced or through the HDMV menu). It will automatically blend smoothly to the main program creating the illusion of a single video stream.

## Example use-cases
- "typesetting" experience, by overlaying the typeset on the unaltered program. The HDMV PGS streams can then be limited to just dialogue.
- "unaltered" experience, by masking the hard-sub of the primary program with the unaltered one.
- Alternate cuts or in-scene variants.
- Bonus features like a storyboard, a director highlighting elements on the main program (by drawing on a tablet)
- Commentaries with the participants (face camera, MST3K-like, ...)

## Usage
Here's an example to generate a secondary video track with ASS typesetting:
```python
from pipelette import PIPelette

# Load the main program (primary)
primary = core.bs.VideoSource("video_master.mov")

# Load subtitle (overlay)
ass_script = "/path/to/typesetting/script.ass"
overlay = core.sub.TextFile(primary, file=ass_script, fontdir='/a/dir/with/fonts/', blend=False)

# ... pick the resampler to your personal taste (processing in 444 is preferred)
primary = core.resize.Bilinear(primary, format=vs.YUV444P8, ...) #specify parameters like chromaloc as needed

# Generate the secondary video track that will blend smoothly on top of primary.
pip_overlay = PIPelette().PIPelette(primary, overlay)

# Convert to 420p for BD
pip_overlay = core.resize.Bilinear(pip_overlay, format=vs.YUV420P8)
pip_overlay.set_output()
```

## Requirements
- VapourSynth
- numpy
- numba

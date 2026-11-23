"""isp: a raw-to-RGB image signal processor built stage by stage.

Conventions used everywhere in this package
-------------------------------------------
* Mosaics are 2-D float arrays aligned so that the top-left site is red (RGGB).
  ``rawio.load_raw`` does the alignment; nothing downstream checks again.
* "Packed" Bayer is (H/2, W/2, 4) with channel order R, G1, G2, B.
* RGB images are (H, W, 3), float32, channel order R, G, B.
* After the ``black_level`` stage, 0 is black and 1 is the saturation level.
"""

__version__ = "0.1.0"

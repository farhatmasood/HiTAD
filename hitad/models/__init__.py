"""
HiTAD Model Architectures
==============================

Three model variants for spinal disorder detection:

- HiTAD (V1): Single-stage YOLOv8-style detector with AreaAttention
  and Distribution Focal Loss head. Uses C2f backbone, PANet neck.

- HiTADV2: Two-stage detector (DDE + MLDR) with RoI-based refinement.
  Stage 1 produces multi-scale detections; Stage 2 refines via RoI pooling.

- HiTADV33: DenseC2f backbone variant with improved normalization
  and area attention at FPN boundaries.
"""

from .hitad import HiTAD
from .hitad_v2 import HiTADV2, build_model
from .hitad_v33 import HiTADV33

__all__ = [
    "HiTAD",
    "HiTADV2",
    "HiTADV33",
    "build_model",
]

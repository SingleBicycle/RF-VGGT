"""VGGT package compatibility helpers."""

import sys as _sys

_sys.modules.setdefault("vggt.vggt", _sys.modules[__name__])
setattr(_sys.modules[__name__], "vggt", _sys.modules[__name__])

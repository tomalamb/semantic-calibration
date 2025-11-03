"""Open Uncertainty - Calibration and uncertainty quantification for LLMs.

This package intentionally keeps `__init__` minimal to avoid circular imports
when modules inside `src/` import each other. Import training scripts or
submodules explicitly (for example `from src import head_temp_training`) when
needed.
"""

__version__ = "0.1.0"

# Export utilities for convenience. Do NOT import training modules here; that
# causes circular imports when `src` is imported from inside its modules.
from . import utils

__all__ = ["utils"]

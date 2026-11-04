# NVFP4 path is pure-torch (no compiled backend), so it is importable without
# the CUDA extension and is exported unconditionally.
from .nvfp4_linear import NVFP4Linear

# The INT4/INT8 quantized layers require the compiled `omniserve_backend`
# extension. Guard them so the package still imports on CPU-only machines.
try:
    from .w8a8_linear import *
    from .w4a8_linear import *
    from .w4a8_moe_linear import *
except ImportError:  # pragma: no cover - omniserve_backend not built
    pass

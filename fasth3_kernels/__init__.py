"""Independent inference kernels; CUDA/CuTe imports are lazy."""
from .attention import sparse_attention
from .nvfp4 import dequantize_nvfp4, linear_nvfp4

__all__ = ["sparse_attention", "dequantize_nvfp4", "linear_nvfp4"]

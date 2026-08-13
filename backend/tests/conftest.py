"""Shared pytest initialization."""

# PyTorch must be initialized before NumPy/scikit-learn on Windows.
# In the current CUDA environment, the reverse order can cause WinError 1114
# while loading torch/lib/c10.dll.
import torch  # noqa: F401, E402
import os

# Keep tests deterministic and on CPU regardless of the host's accelerators.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

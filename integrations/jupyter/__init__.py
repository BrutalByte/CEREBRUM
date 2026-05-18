"""CEREBRUM Jupyter integration — magic commands and inline trace visualization."""
from .cerebrum_magic import display_trace, load_ipython_extension, unload_ipython_extension

__all__ = ["display_trace", "load_ipython_extension", "unload_ipython_extension"]

"""Process-local compatibility shim for EmbodiedGen V2's Gin namespace includes.

Gin 0.5.0 treats Infinigen's namespace-package config folders as importable packages and can
raise ``TypeError`` before trying the next filesystem search prefix.  This shim removes only
that broken resource-package reader and registers the frozen Infinigen checkout as a normal
filesystem prefix.  No EmbodiedGen or Infinigen source file is modified.
"""

from __future__ import annotations

import os
from pathlib import Path

root_value = os.environ.get("KINO_EMBODIEDGEN_INFINIGEN_ROOT")
if root_value:
    root = Path(root_value).resolve()
    if not (root / "infinigen_examples/configs_indoor/base_indoors.gin").is_file():
        raise RuntimeError(f"invalid frozen Infinigen config root: {root}")
    import gin
    from gin import config

    # Gin's built-in ``open``/``isfile`` reader is sufficient for Infinigen's checked-out
    # configs.  The additional resource reader dereferences ``spec.origin`` for namespace
    # packages even when it is ``None`` and crashes before the filesystem fallback is tried.
    config._FILE_READERS[:] = [
        (reader, exists)
        for reader, exists in config._FILE_READERS
        if getattr(exists, "__module__", "") != "gin.resource_reader"
    ]
    if not config._FILE_READERS:
        raise RuntimeError("EmbodiedGen Gin compatibility shim removed every file reader")
    gin.add_config_file_search_path(str(root))

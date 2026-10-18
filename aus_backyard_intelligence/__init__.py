"""AUS Backyard Intelligence — Australia-wide backyard opportunity platform.

Pool/solar detection, state-aware compliance, and contractor lead generation
across all Australian states and territories.
"""

import os as _os

__all__ = ["BackyardDemoApp", "create_app", "detect_qgis_process", "qgis_status", "serve"]
__version__ = "0.2.0"

# Env-var compatibility: the public prefix is ABI_* (AUS Backyard Intelligence);
# internal reads still use the legacy QLDR_* names. Mirror ABI_* onto QLDR_* at
# import time so either prefix works. QLDR_* set explicitly always wins.
for _key, _value in list(_os.environ.items()):
    if _key.startswith("ABI_"):
        _os.environ.setdefault("QLDR_" + _key[len("ABI_"):], _value)


def __getattr__(name):
    if name in {"BackyardDemoApp", "create_app", "serve"}:
        from .app import BackyardDemoApp, create_app, serve

        return {
            "BackyardDemoApp": BackyardDemoApp,
            "create_app": create_app,
            "serve": serve,
        }[name]
    if name == "export_ranked_map":
        from .map_export import export_ranked_map

        return export_ranked_map
    if name in {"detect_qgis_process", "qgis_status"}:
        from .qgis import detect_qgis_process, qgis_status

        return {
            "detect_qgis_process": detect_qgis_process,
            "qgis_status": qgis_status,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _candidate_paths() -> list[Path]:
    env_path = os.environ.get("QGIS_PROCESS_PATH")
    candidates = [Path(env_path)] if env_path else []
    for root in _qgis_install_roots():
        candidates.extend([
            root / "bin" / "qgis_process-qgis-ltr.bat",
            root / "bin" / "qgis_process.bat",
            root / "apps" / "qgis-ltr" / "bin" / "qgis_process.exe",
            root / "apps" / "qgis" / "bin" / "qgis_process.exe",
        ])
    return [path for path in candidates if path is not None]


def _desktop_candidate_paths() -> list[Path]:
    env_path = os.environ.get("QGIS_DESKTOP_PATH")
    candidates = [Path(env_path)] if env_path else []
    for root in _qgis_install_roots():
        candidates.extend([
            root / "bin" / "qgis-ltr.bat",
            root / "bin" / "qgis.bat",
            root / "apps" / "qgis-ltr" / "bin" / "qgis-bin.exe",
            root / "apps" / "qgis" / "bin" / "qgis-bin.exe",
        ])
    return [path for path in candidates if _path_exists(path)]


def _qgis_install_roots() -> list[Path]:
    env_roots = [
        os.environ.get("QGIS_INSTALL_ROOT"),
        os.environ.get("QGIS_ROOT"),
        os.environ.get("OSGEO4W_ROOT"),
    ]
    roots: list[Path] = [Path(value) for value in env_roots if value]
    search_roots = [
        Path.home() / "AppData" / "Local" / "Programs",
        Path("C:/Program Files"),
    ]
    for base in search_roots:
        try:
            roots.extend(sorted(path for path in base.glob("QGIS*") if path.is_dir()))
        except OSError:
            pass
    seen: set[str] = set()
    unique_roots: list[Path] = []
    for root in roots:
        key = str(root).lower()
        if key not in seen:
            seen.add(key)
            unique_roots.append(root)
    return unique_roots


def detect_qgis_process() -> Path | None:
    for candidate in _candidate_paths():
        if _path_exists(candidate):
            return candidate
    return None


def detect_qgis_desktop() -> Path | None:
    candidates = _desktop_candidate_paths()
    return candidates[0] if candidates else None


def _qgis_command(*args: str) -> list[str]:
    qgis_path = detect_qgis_process()
    if qgis_path is None:
        raise FileNotFoundError("QGIS process launcher was not found.")
    if qgis_path.suffix.lower() == ".bat":
        return ["cmd", "/c", str(qgis_path), *args]
    return [str(qgis_path), *args]


def run_qgis(*args: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _qgis_command(*args),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def open_in_qgis(path: str | Path) -> dict[str, Any]:
    qgis_path = detect_qgis_desktop()
    if qgis_path is None:
        raise FileNotFoundError("QGIS desktop launcher was not found.")
    target = str(Path(path).resolve())
    if qgis_path.suffix.lower() == ".bat":
        process = subprocess.Popen(["cmd", "/c", "start", "", str(qgis_path), target])
    else:
        process = subprocess.Popen([str(qgis_path), target])
    return {"launcher": str(qgis_path), "target": target, "pid": process.pid}


def list_algorithms(limit: int = 20) -> list[str]:
    completed = run_qgis("list", timeout=240)
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    algorithms = [line for line in lines if ":" in line and not line.lower().startswith("provider")]
    return algorithms[: max(limit, 1)]


def qgis_status(*, probe_version: bool = True) -> dict[str, Any]:
    qgis_path = detect_qgis_process()
    desktop_path = detect_qgis_desktop()
    if qgis_path is None:
        return {
            "configured": False,
            "path": None,
            "desktop_path": str(desktop_path) if desktop_path else None,
            "version": None,
        }
    version = None
    if probe_version:
        try:
            completed = run_qgis("--version", timeout=8)
            version = (completed.stdout or completed.stderr).strip() or None
        except Exception:
            version = None
    return {
        "configured": True,
        "path": str(qgis_path),
        "desktop_path": str(desktop_path) if desktop_path else None,
        "version": version,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check QGIS availability for the backyard recommender project.")
    parser.add_argument("--check", action="store_true", help="Print the detected QGIS status as JSON.")
    parser.add_argument("--list-algorithms", action="store_true", help="List a sample of available QGIS processing algorithms.")
    parser.add_argument("--limit", type=int, default=20, help="How many algorithm names to print when listing algorithms.")
    parser.add_argument("--open", help="Open a file in QGIS Desktop.")
    args = parser.parse_args(argv)
    if args.open:
        print(json.dumps(open_in_qgis(args.open), indent=2))
        return
    if args.list_algorithms:
        print(json.dumps({"algorithms": list_algorithms(limit=args.limit)}, indent=2))
        return
    print(json.dumps(qgis_status(), indent=2))


if __name__ == "__main__":
    main()

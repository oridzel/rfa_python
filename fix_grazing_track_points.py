#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import py_compile
import shutil

MARKER = "GRAZING_ANALYTIC_SAMPLE_VOXEL_BYPASS_V1"

OLD_PBEFORE = '''                    if len(traj) >= 2:
                        _sample_hit["p_before"] = np.asarray(
                            traj[-2], dtype=float
                        ).copy()
'''

NEW_PBEFORE = '''                    if traj is not None and len(traj) >= 2:
                        _sample_hit["p_before"] = np.asarray(
                            traj[-2], dtype=float
                        ).copy()
'''

OLD_RETURN = '''                    return {
                        "reason": "hit_sample",
                        "hit_info": _sample_hit,
                        "traj": np.asarray(traj),
                        "vel": np.asarray(vel),
                        "steps": step,
                        "grid_events": grid_events,
                    }
'''

NEW_RETURN = '''                    return {
                        "reason": "hit_sample",
                        "hit_info": _sample_hit,
                        "traj": (
                            np.asarray(traj, dtype=float)
                            if traj is not None
                            else None
                        ),
                        "vel": (
                            np.asarray(vel, dtype=float)
                            if vel is not None
                            else None
                        ),
                        "steps": step,
                        "grid_events": grid_events,
                        "events": grid_events,
                    }
'''


def patch(path: Path) -> None:
    text = path.read_text()

    if MARKER not in text:
        raise RuntimeError(
            f"{MARKER} not found. This file does not appear to contain "
            "the emitted grazing patch."
        )

    marker_pos = text.index(MARKER)
    tail = text[marker_pos:]

    changed = False

    if OLD_PBEFORE in tail:
        tail = tail.replace(OLD_PBEFORE, NEW_PBEFORE, 1)
        changed = True

    if OLD_RETURN in tail:
        tail = tail.replace(OLD_RETURN, NEW_RETURN, 1)
        changed = True

    if not changed:
        local = tail[:8000]
        if "if len(traj) >= 2:" in local:
            raise RuntimeError(
                "Found an unsafe len(traj) pattern, but the surrounding code "
                "differs from the expected patch. Upload trajectories.py so it "
                "can be patched exactly."
            )
        print("The grazing block already appears track_points-safe.")
    else:
        backup = path.with_suffix(path.suffix + ".pre_grazing_track_fix.bak")
        if not backup.exists():
            shutil.copy2(path, backup)

        text = text[:marker_pos] + tail
        path.write_text(text)
        print("Patched grazing recovery for track_points=False.")
        print(f"Backup: {backup}")

    py_compile.compile(str(path), doraise=True)
    print(f"Syntax compilation successful: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="rfa_model/trajectories.py")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.is_file():
        raise SystemExit(f"File not found: {path}")

    patch(path)


if __name__ == "__main__":
    main()

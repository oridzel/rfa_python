#!/usr/bin/env python3
"""Patch rfa_model/trajectories.py for grazing analytic-sample voxel artifacts.

Usage:
    python apply_emitted_grazing_voxel_fix.py rfa_model/trajectories.py

The patch is intentionally narrow. Inside integrate_one_electron(), a fixed
voxel owned by the sample is not treated as a collision when the point is still
on the analytic vacuum side of the finite sample plane. The ordinary field
integrator continues, preserving electrostatic turning of grazing BSEs. A true
analytic-plane return remains terminal.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import py_compile
import shutil

MARKER = "GRAZING_ANALYTIC_SAMPLE_VOXEL_BYPASS_V1"

INSERT = r'''
        # GRAZING_ANALYTIC_SAMPLE_VOXEL_BYPASS_V1
        # ---------------------------------------------------------------
        # The voxelized sample exists to impose the electrostatic boundary;
        # the finite analytic sample plane is the physical collision surface.
        # At grazing incidence the voxel staircase can protrude a long way
        # into physical vacuum along an emitted-electron ray. Do NOT terminate
        # such a trajectory merely because its nearest field node is a sample
        # fixed voxel.
        #
        # Unlike the high-energy primary artifact helper, we deliberately do
        # not jump geometrically through many millimeters here. A grazing BSE
        # can have tiny normal kinetic energy and can be physically turned by
        # the sample-bias field. Therefore the ordinary field integrator keeps
        # advancing while only the collision classification of an analytically
        # vacuum sample voxel is neutralized.
        if (
            sample_plane_return
            and sample_geometry is not None
            and cls.get("status") == "hit_fixed"
        ):
            _sample_owner_id = cls.get("owner_id", None)
            _sample_owner = fixed_owner_name(_sample_owner_id, field=field)

            if _sample_owner == "sample":
                _C_sample = np.asarray(sample_geometry["center"], dtype=float)
                _n_sample = unit(
                    np.asarray(sample_geometry["normal"], dtype=float)
                )
                _signed_sample = float(np.dot(p - _C_sample, _n_sample))
                _projected_sample = p - _signed_sample * _n_sample
                _projected_in_bounds = sample_point_is_in_bounds(
                    _projected_sample, sample_geometry
                )
                _travel_from_launch = float(
                    np.linalg.norm(
                        _projected_sample - np.asarray(p0, dtype=float)
                    )
                )

                # If numerical stepping has already placed the electron on the
                # solid side of the true finite face, recover the physical
                # analytic-plane return instead of calling it a voxel hit.
                if (
                    _signed_sample <= 0.0
                    and _projected_in_bounds
                    and _travel_from_launch >= float(min_sample_return_distance)
                ):
                    _, _sample_u, _sample_v = sample_plane_coordinates(
                        _projected_sample, sample_geometry
                    )
                    _sample_hit = {
                        "kind": "sample_plane_recovered_from_voxel",
                        "location": _projected_sample.copy(),
                        "distance": float(
                            np.linalg.norm(_projected_sample - p)
                        ),
                        "owner": "sample",
                        "owner_name": "sample",
                        "owner_id": _sample_owner_id,
                        "normal": _n_sample.copy(),
                        "sample_u": _sample_u,
                        "sample_v": _sample_v,
                        "sample_theta_deg": sample_geometry.get(
                            "theta_deg", np.nan
                        ),
                        "grid_classification": dict(cls),
                        "sample_voxel_artifact_recovered": True,
                        "sample_voxel_artifact_signed_distance_m": (
                            _signed_sample
                        ),
                        "KE_hit_eV": kinetic_energy_eV_from_velocity(v),
                        "v_in": np.asarray(v, dtype=float).copy(),
                    }
                    if len(traj) >= 2:
                        _sample_hit["p_before"] = np.asarray(
                            traj[-2], dtype=float
                        ).copy()

                    return {
                        "reason": "hit_sample",
                        "hit_info": _sample_hit,
                        "traj": np.asarray(traj),
                        "vel": np.asarray(vel),
                        "steps": step,
                        "grid_events": grid_events,
                    }

                # Still in physical vacuum (or outside the finite sample face):
                # ignore only the sample-voxel collision label. The field
                # integration, STL tests, analytic sample-plane tests, grid
                # crossings, and all other physics continue normally below.
                cls = dict(cls)
                cls.update({
                    "status": "free",
                    "ignored_fixed_owner": "sample",
                    "sample_voxel_artifact_ignored": True,
                    "sample_voxel_artifact_signed_distance_m": (
                        _signed_sample
                    ),
                    "sample_voxel_artifact_projected_in_bounds": bool(
                        _projected_in_bounds
                    ),
                })
'''


def patch(path: Path) -> Path:
    text = path.read_text()
    if MARKER in text:
        print(f"Already patched: {path}")
        return path

    fn = text.find("def integrate_one_electron(")
    if fn < 0:
        raise RuntimeError("Could not find integrate_one_electron()")

    needle = "        cls = classify_effective_vacuum_point(p, field)\n"
    pos = text.find(needle, fn)
    if pos < 0:
        raise RuntimeError(
            "Could not find classify_effective_vacuum_point(p, field) inside "
            "integrate_one_electron(). Upload trajectories.py if your branch differs."
        )

    insert_at = pos + len(needle)
    patched = text[:insert_at] + INSERT + text[insert_at:]

    backup = path.with_suffix(path.suffix + ".pre_grazing_emit_fix.bak")
    if not backup.exists():
        shutil.copy2(path, backup)

    path.write_text(patched)
    py_compile.compile(str(path), doraise=True)
    print(f"Patched and compiled: {path}")
    print(f"Backup: {backup}")
    return path


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

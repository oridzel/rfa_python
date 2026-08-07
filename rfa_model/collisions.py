"""
collisions.py

STL collision helpers, analytic spherical grid/collector crossings,
grid/opening logic, and analytic sample-plane intersections.
"""

from __future__ import annotations

import warnings

import numpy as np
import trimesh
from .constants import COLLECTOR_OPENING_ALPHA_DEG


OWNER_ID_BY_NAME = {
    "sample": 1,
    "holder": 2,
    "receiver": 3,
    "rod": 4,

    "g1frame": 5,
    "g2frame": 6,
    "g3frame": 7,

    "g1_low_frame": 5,
    "g1_upper_frame": 5,
    "g2_low_frame": 6,
    "g2_upper_frame": 6,
    "g3_low_frame": 7,
    "g3_upper_frame": 7,

    "drifttube": 8,

    "g1_shell": 9,
    "g2_shell": 10,
    "g3_shell": 11,
    "collector_shell": 12,
}


def _is_finite_vec(v, n=3) -> bool:
    """
    Return True if v is a finite vector of length n.
    """
    v = np.asarray(v, dtype=float)

    return (
        v.shape == (n,)
        and np.all(np.isfinite(v))
    )


def canonical_collision_owner_name(name):
    """
    Keep detailed grid-frame names for geometry, but normalize aliases if needed.
    """
    if name is None:
        return None

    name = str(name)

    aliases = {
        "grid1": "g1_shell",
        "grid2": "g2_shell",
        "grid3": "g3_shell",
        "collector": "collector_shell",
    }

    return aliases.get(name, name)


def add_owner_metadata(hit: dict, owner_name: str | None = None) -> dict:
    """
    Add explicit owner_name and owner_id fields to a hit dictionary.

    Keeps the older 'owner' key for backward compatibility.
    """
    if hit is None:
        return None

    if owner_name is None:
        owner_name = hit.get("owner_name", None)
    if owner_name is None:
        owner_name = hit.get("owner", None)

    owner_name = canonical_collision_owner_name(owner_name)

    hit["owner"] = owner_name
    hit["owner_name"] = owner_name
    hit["owner_id"] = OWNER_ID_BY_NAME.get(owner_name, None)

    return hit


# ============================================================
# STL collision mesh and intersector
# ============================================================

def combine_labeled_meshes(meshes_for_collision: dict[str, trimesh.Trimesh]):
    """
    Combine multiple named meshes into one trimesh and preserve face-owner labels.

    Parameters
    ----------
    meshes_for_collision:
        Dict mapping part name -> trimesh.

    Returns
    -------
    combined_mesh:
        Single combined trimesh.
    face_owner:
        Object array where face_owner[i] is the part name for face i.
    """
    vertices_all = []
    faces_all = []
    face_owner = []

    vertex_offset = 0

    for name, mesh in meshes_for_collision.items():
        V = mesh.vertices.copy()
        F = mesh.faces.copy() + vertex_offset

        vertices_all.append(V)
        faces_all.append(F)
        face_owner.extend([name] * len(mesh.faces))

        vertex_offset += len(V)

    vertices_all = np.vstack(vertices_all)
    faces_all = np.vstack(faces_all)
    face_owner = np.array(face_owner, dtype=object)

    combined_mesh = trimesh.Trimesh(
        vertices=vertices_all,
        faces=faces_all,
        process=False,
    )

    return combined_mesh, face_owner


def _make_ray_intersector(collision_mesh, prefer_embree: bool = True):
    """
    Build the fastest available trimesh ray intersector for a mesh.

    trimesh ships two backends with the same API:

      trimesh.ray.ray_pyembree.RayMeshIntersector
          Wraps Intel Embree (pip install embreex). Benchmarked on a
          ~5k-triangle RFA-scale collision mesh: ~16k rays/s for single-ray
          calls (as used by first_segment_hit) versus ~3.4k rays/s for the
          fallback, i.e. about 5x. If the integrator is ever batched so that
          many electrons are stepped together, the same backend reaches
          ~5M rays/s -- three orders of magnitude -- because Embree amortises
          its BVH traversal over the whole ray packet.

      trimesh.ray.ray_triangle.RayMeshIntersector
          Pure NumPy plus an rtree bounding-volume lookup. Always present, but
          it re-enters Python for every ray.

    Both return identical geometric results, so this is purely a performance
    switch: no physics changes, nothing to re-validate.

    Set prefer_embree=False to force the fallback, e.g. when checking that an
    unexpected result is not a backend artefact.
    """
    backend = "ray_triangle (pure NumPy fallback)"

    if prefer_embree:
        try:
            from trimesh.ray.ray_pyembree import RayMeshIntersector as _Embree

            intersector = _Embree(collision_mesh)
            backend = "pyembree/embreex"

            return intersector, backend

        except Exception as exc:
            # embreex not installed, or failed to build an acceleration
            # structure for this mesh. Fall through rather than crash: the
            # NumPy backend gives the same answers, just slower.
            warnings.warn(
                "Embree ray backend unavailable "
                f"({type(exc).__name__}: {exc}); falling back to the pure-NumPy "
                "intersector, which is roughly 5x slower for this workload. "
                "Install it with:  pip install embreex",
                RuntimeWarning,
                stacklevel=2,
            )

    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(
        collision_mesh
    )

    return intersector, backend


def build_stl_intersector(
    meshes_for_collision: dict[str, trimesh.Trimesh],
    prefer_embree: bool = True,
    verbose: bool = True,
):
    """
    Build a trimesh ray intersector for a labeled collision mesh.

    Uses the Embree backend when available; see _make_ray_intersector.
    The chosen backend is recorded on the returned intersector as
    ``intersector.rfa_backend`` so a run summary can log which one was used.
    """
    collision_mesh, face_owner = combine_labeled_meshes(meshes_for_collision)

    intersector, backend = _make_ray_intersector(
        collision_mesh,
        prefer_embree=prefer_embree,
    )

    try:
        intersector.rfa_backend = backend
    except AttributeError:
        # Some backends use __slots__; the label is a convenience, not a
        # requirement, so a failure here must not break the build.
        pass

    if verbose:
        print(
            f"[collisions] ray backend: {backend} "
            f"({len(collision_mesh.faces):,} triangles)"
        )

    return collision_mesh, face_owner, intersector


def first_segment_hit(
    p0,
    p1,
    intersector,
    face_owner,
    collision_mesh,
    eps: float = 1e-12,
    min_distance: float = 1e-9,
):
    """
    Find first STL hit along finite segment p0 -> p1.

    Returns None if no valid hit occurs within the segment.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)

    if not _is_finite_vec(p0) or not _is_finite_vec(p1):
        return None

    seg = p1 - p0
    seg_len = np.linalg.norm(seg)

    if (not np.isfinite(seg_len)) or seg_len <= eps:
        return None

    direction = seg / seg_len

    if not _is_finite_vec(direction):
        return None

    try:
        locations, ray_ids, tri_ids = intersector.intersects_location(
            ray_origins=p0.reshape(1, 3),
            ray_directions=direction.reshape(1, 3),
            multiple_hits=False,
        )
    except Exception:
        # Avoid letting rare rtree/trimesh failures kill a whole batch.
        # Usually caused by invalid numerical ray state.
        return None

    if len(locations) == 0:
        return None

    location = locations[0]

    if not _is_finite_vec(location):
        return None

    tri_id = int(tri_ids[0])

    distance = np.linalg.norm(location - p0)

    if (not np.isfinite(distance)) or distance <= min_distance:
        return None

    if distance > seg_len + eps:
        return None

    owner = face_owner[tri_id]
    normal = collision_mesh.face_normals[tri_id]

    hit = {
        "kind": "stl",
        "location": location,
        "distance": distance,
        "face_id": tri_id,
        "owner": owner,
        "normal": normal,
    }

    # If you added add_owner_metadata earlier, use it:
    if "add_owner_metadata" in globals():
        hit = add_owner_metadata(hit, owner_name=owner)

    return hit


# ============================================================
# STL broad-phase bounding boxes
# ============================================================

def build_stl_bounding_boxes(
    meshes_for_collision: dict[str, trimesh.Trimesh],
    padding: float = 1.0e-3,
):
    """
    Build padded axis-aligned bounding boxes for broad-phase STL checks.
    """
    boxes = []

    for name, mesh in meshes_for_collision.items():
        bounds = mesh.bounds.copy()

        bounds[0, :] -= padding
        bounds[1, :] += padding

        boxes.append({
            "name": name,
            "bounds": bounds,
        })

    return boxes


def segment_intersects_aabb(
    p0,
    p1,
    bounds,
    eps: float = 1e-15,
) -> bool:
    """
    Check whether segment p0 -> p1 intersects an axis-aligned bounding box.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    bounds = np.asarray(bounds, dtype=float)

    if (
        not _is_finite_vec(p0)
        or not _is_finite_vec(p1)
        or bounds.shape != (2, 3)
        or not np.all(np.isfinite(bounds))
    ):
        return False

    d = p1 - p0

    if not _is_finite_vec(d):
        return False

    tmin = 0.0
    tmax = 1.0

    for ax in range(3):
        if abs(d[ax]) < eps:
            if p0[ax] < bounds[0, ax] or p0[ax] > bounds[1, ax]:
                return False
        else:
            inv_d = 1.0 / d[ax]

            t1 = (bounds[0, ax] - p0[ax]) * inv_d
            t2 = (bounds[1, ax] - p0[ax]) * inv_d

            if not np.isfinite(t1) or not np.isfinite(t2):
                return False

            if t1 > t2:
                t1, t2 = t2, t1

            tmin = max(tmin, t1)
            tmax = min(tmax, t2)

            if tmin > tmax:
                return False

    return True


def segment_near_any_stl_box(p0, p1, stl_boxes) -> bool:
    """
    Broad-phase test: return True if segment intersects any padded STL box.
    """
    for box in stl_boxes:
        if segment_intersects_aabb(p0, p1, box["bounds"]):
            return True

    return False


# ============================================================
# Analytic spherical grid / collector surfaces
# ============================================================

def first_sphere_segment_crossing(
    p0,
    p1,
    radius: float,
    name: str,
    center=np.zeros(3),
    eps: float = 1e-12,
):
    """
    First crossing of a sphere by finite segment p0 -> p1.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    center = np.asarray(center, dtype=float)

    if (
        not _is_finite_vec(p0)
        or not _is_finite_vec(p1)
        or not _is_finite_vec(center)
        or not np.isfinite(radius)
    ):
        return None

    p0_shift = p0 - center
    p1_shift = p1 - center

    d = p1_shift - p0_shift

    a = np.dot(d, d)
    b = 2.0 * np.dot(p0_shift, d)
    c = np.dot(p0_shift, p0_shift) - radius**2

    if not np.isfinite(a) or not np.isfinite(b) or not np.isfinite(c):
        return None

    disc = b * b - 4.0 * a * c

    if (not np.isfinite(disc)) or disc < 0 or a <= eps:
        return None

    sqrt_disc = np.sqrt(disc)

    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    candidates = [
        t for t in (t1, t2)
        if np.isfinite(t) and eps < t <= 1.0 + eps
    ]

    if not candidates:
        return None

    t = min(candidates)

    location = (p0_shift + t * d) + center

    if not _is_finite_vec(location):
        return None

    normal = location - center
    normal_norm = np.linalg.norm(normal)

    if (not np.isfinite(normal_norm)) or normal_norm <= eps:
        return None

    normal = normal / normal_norm

    hit = {
        "kind": "sphere",
        "location": location,
        "distance": np.linalg.norm(t * d),
        "t": t,
        "owner": name,
        "normal": normal,
    }

    if "add_owner_metadata" in globals():
        hit = add_owner_metadata(hit, owner_name=name)

    return hit


# ============================================================
# Grid / collector openings
# ============================================================

def compute_rod_opening_geometry(
    rod_mesh,
    shell_radii: dict,
    clearance_m: float = 1.0e-3,
    band_thickness_m: float = 2.0e-3,
    n_samples: int = 200_000,
    seed: int = 0,
):
    """
    Precompute the ROD's actual local footprint (as seen by each analytic
    shell) directly from the rod STL mesh, replacing the fixed
    radius=11mm-circle-at-origin approximation that ignores the rod's real
    off-axis position and irregular cross-section (e.g. near the sample
    receiver attachment).

    For each shell radius R in shell_radii, area-weighted surface points are
    sampled from the rod mesh (NOT just mesh vertices - a coarse mesh can
    easily have zero vertices in a thin radial band even though a face
    clearly crosses it) and the subset within +/- band_thickness_m of R is
    used to fit a local bounding circle in the (x, y) plane, inflated by
    clearance_m to leave physical clearance between the rod and the grid
    wire / collector material around it.

    Call this ONCE at setup time (alongside computing R_g1/R_g2/R_g3/R_col)
    and store the result as field["rod_openings"]; is_in_rod_opening() then
    just does a fast lookup + point-in-circle test per trajectory step.

    Parameters
    ----------
    rod_mesh:
        The rod's trimesh.Trimesh (e.g. collision_meshes_emit["rod"]).
    shell_radii:
        Dict like {"g1_shell": R_g1, "g2_shell": R_g2, "g3_shell": R_g3,
        "collector_shell": R_col}.
    clearance_m:
        Extra radius added on top of the rod's measured footprint, to
        represent manufacturing/assembly clearance between the rod and
        the surrounding mesh/collector (default 1 mm, per instrument spec).
    band_thickness_m:
        Half-thickness of the radial shell used to collect sample points
        for a given shell radius.
    n_samples:
        Number of area-weighted surface samples drawn from the whole rod
        mesh. Larger values increase the chance of finding thin/rare
        crossings (e.g. at large shell radii where only a small part of
        the rod's length is nearby).

    Returns
    -------
    dict[shell_name] -> {"center_xy": (x0, y0), "z_ref": z0, "radius": r_eff}
        or None if no sampled points were found near that shell radius
        (i.e. the rod does not appear to cross that shell at all).
    """
    import trimesh

    rng = np.random.default_rng(seed)
    pts, _ = trimesh.sample.sample_surface(rod_mesh, n_samples, seed=rng)
    pts = np.asarray(pts, dtype=float)
    r = np.linalg.norm(pts, axis=1)

    openings = {}
    for name, R in shell_radii.items():
        mask = np.abs(r - float(R)) < band_thickness_m
        band_pts = pts[mask]

        if len(band_pts) == 0:
            openings[name] = None
            continue

        x0 = float(np.mean(band_pts[:, 0]))
        y0 = float(np.mean(band_pts[:, 1]))
        z0 = float(np.mean(band_pts[:, 2]))

        d = np.sqrt(
            (band_pts[:, 0] - x0) ** 2 + (band_pts[:, 1] - y0) ** 2
        )
        r_eff = float(d.max()) + float(clearance_m)

        openings[name] = {
            "center_xy": (x0, y0),
            "z_ref": z0,
            "radius": r_eff,
            "n_points": int(len(band_pts)),
        }

    return openings


def compute_drifttube_bore_geometry(
    dt_mesh,
    x_domain_max: float,
    band_thickness_m: float = 1.0e-3,
    n_samples: int = 200_000,
    probe_fractions=(0.30, 0.50, 0.70),
    seed: int = 0,
):
    """
    Derive the drift tube's real bore radius, axis offset, and exit plane
    directly from its STL mesh.

    This replaces the fixed on-axis 5.6 mm circle that
    is_in_drift_tube_aperture() and trajectories.is_drifttube_escape_candidate()
    both fell back on. Now that the drift tube is real collision geometry, the
    analytic escape aperture and the STL must describe the same bore, otherwise
    an electron can be judged "escaped" through a nominal circle that the real
    tube wall would actually have intercepted (or vice versa).

    Method (mirrors compute_rod_opening_geometry): area-weighted surface points
    are sampled from the whole mesh, then thin YZ slabs are taken at several
    interior x positions. Mesh VERTICES are not used, because a coarse tube mesh
    can easily have no vertices inside a thin slab even though faces clearly
    cross it. Interior probe planes are used rather than the ends so that end
    caps or flanges cannot contaminate the inner-wall radius estimate.

    In each slab the radii cluster at the inner wall and the outer wall. The
    inner cluster is isolated with a midpoint threshold, its centroid gives the
    bore axis offset, and the median of its radii gives the bore radius. The
    median is robust to the handful of stray points a triangulated fillet or
    chamfer contributes.

    Parameters
    ----------
    dt_mesh:
        The drift tube's trimesh.Trimesh, already aligned into RFA coordinates
        (e.g. collision_meshes_emit["drifttube"]). If None, returns None.
    x_domain_max:
        Largest x of the field/update domain, i.e. field["x"][-1]. The escape
        plane is the smaller of the tube's own +X end and this value, since an
        electron cannot be tracked past the domain edge.
    band_thickness_m:
        Half-thickness of each YZ slab used to sample the bore.
    n_samples:
        Number of area-weighted surface samples over the whole mesh.
    probe_fractions:
        Fractional positions along the tube's x-span at which to probe the bore.
        The median across probes is returned, so a locally odd slab cannot skew
        the result.

    Returns
    -------
    dict with:
        center_yz    (y0, z0) bore axis offset from the RFA axis, metres
        radius       bore radius, metres
        x_min, x_max the tube's own x extent, metres
        x_exit       escape plane, min(x_max, x_domain_max), metres
        n_probes     how many probe planes yielded a usable inner-wall cluster
        source       provenance string for logging
    or None if dt_mesh is None or no probe plane yielded a usable cluster,
    in which case callers fall back to the legacy on-axis circle.
    """
    if dt_mesh is None:
        return None

    import trimesh

    rng = np.random.default_rng(seed)
    pts, _ = trimesh.sample.sample_surface(dt_mesh, n_samples, seed=rng)
    pts = np.asarray(pts, dtype=float)

    x_lo = float(pts[:, 0].min())
    x_hi = float(pts[:, 0].max())
    span = x_hi - x_lo

    if not np.isfinite(span) or span <= 0.0:
        return None

    radii = []
    centers = []

    for frac in probe_fractions:
        x_probe = x_lo + float(frac) * span
        slab = pts[np.abs(pts[:, 0] - x_probe) < band_thickness_m]

        if len(slab) < 50:
            continue

        # Provisional centre, then isolate the inner-wall cluster.
        y0 = float(np.mean(slab[:, 1]))
        z0 = float(np.mean(slab[:, 2]))
        rho = np.hypot(slab[:, 1] - y0, slab[:, 2] - z0)

        r_in_guess = float(np.percentile(rho, 2.0))
        r_out_guess = float(np.percentile(rho, 98.0))

        if not np.isfinite(r_in_guess) or r_out_guess <= 0.0:
            continue

        inner = slab[rho < 0.5 * (r_in_guess + r_out_guess)]

        if len(inner) < 20:
            continue

        # Refine the axis on the inner wall only, then measure the bore.
        y1 = float(np.mean(inner[:, 1]))
        z1 = float(np.mean(inner[:, 2]))
        rho_inner = np.hypot(inner[:, 1] - y1, inner[:, 2] - z1)

        radii.append(float(np.median(rho_inner)))
        centers.append((y1, z1))

    if not radii:
        return None

    radius = float(np.median(radii))
    y_c = float(np.median([c[0] for c in centers]))
    z_c = float(np.median([c[1] for c in centers]))

    return {
        "center_yz": (y_c, z_c),
        "radius": radius,
        "x_min": x_lo,
        "x_max": x_hi,
        "x_exit": float(min(x_hi, float(x_domain_max))),
        "n_probes": int(len(radii)),
        "source": "drift tube STL, area-weighted interior slabs",
    }


def drifttube_bore_from_field(field: dict | None):
    """
    Return field["drifttube_bore"] if it looks usable, else None.

    Centralises the "do we have real bore geometry?" test so the aperture check
    and the escape check cannot disagree about which representation is active.
    """
    if not isinstance(field, dict):
        return None

    bore = field.get("drifttube_bore", None)

    if not isinstance(bore, dict):
        return None

    try:
        radius = float(bore["radius"])
        y0, z0 = bore["center_yz"]
        float(y0)
        float(z0)
        float(bore["x_exit"])
    except (KeyError, TypeError, ValueError):
        return None

    if not np.isfinite(radius) or radius <= 0.0:
        return None

    return bore


def is_in_drift_tube_aperture(
    p,
    field: dict,
    r_hole: float | None = None,
) -> bool:
    """
    Check whether point on a spherical surface lies in the drift-tube aperture.

    The drift-tube aperture is near +X and has radius r_hole in the YZ plane.

    Preferred path: field["drifttube_bore"], derived from the real drift-tube
    STL by compute_drifttube_bore_geometry(). That supplies both the bore radius
    and its actual offset from the RFA axis, so an off-axis tube is handled the
    same way is_in_rod_opening() already handles the off-axis rod.

    Legacy path: field["drifttube_aperture_radius"], falling back to 5.6 mm,
    treated as a circle centred exactly on the axis. Kept so that setups which
    have not yet called compute_drifttube_bore_geometry() still run unchanged.

    An explicit r_hole argument always wins, and is always interpreted on-axis.

    trajectories.is_drifttube_escape_candidate() resolves the geometry through
    the same helper, so both checks always describe the same physical bore.
    """
    p = np.asarray(p, dtype=float)

    if r_hole is None:
        bore = drifttube_bore_from_field(field)

        if bore is not None:
            y0, z0 = bore["center_yz"]
            rho_yz = float(np.hypot(p[1] - float(y0), p[2] - float(z0)))

            return (p[0] > 0.0) and (rho_yz <= float(bore["radius"]))

        r_hole = float((field or {}).get("drifttube_aperture_radius", 0.0056))

    rho_yz = np.sqrt(p[1]**2 + p[2]**2)

    return (p[0] > 0.0) and (rho_yz <= r_hole)


def is_in_rod_opening(
    p,
    owner: str,
    field: dict,
) -> bool:
    """
    Check whether point p, known to lie on analytic shell `owner`, falls
    within that shell's rod opening.

    Uses field["rod_openings"][owner] - precomputed once from the real rod
    STL mesh via compute_rod_opening_geometry() - rather than a single
    fixed radius=11mm circle centered at the origin. The real rod is
    measurably off-axis (by several mm at some shells) and not perfectly
    circular, so a shared fixed circle either clips real rod-bound
    trajectories onto the grid/collector, or lets grid/collector-bound
    trajectories slip through as if the rod were there - both of which look
    exactly like "rod current is structurally too low, regardless of any
    yield/transparency parameter", which is the symptom this was chasing.

    Falls back to False (no opening - treat as solid) for a shell where no
    rod-mesh samples were found nearby, and to the old fixed-circle
    approximation only if field["rod_openings"] itself is entirely absent
    (e.g. not yet computed for this field), to avoid silently breaking
    existing setups.
    """
    p = np.asarray(p, dtype=float)

    rod_openings = (field or {}).get("rod_openings", None)

    if rod_openings is None:
        # Backward-compatible fallback: old fixed-circle approximation.
        r_rod = float((field or {}).get("rod_opening_radius", 0.011))
        rho_xy = np.sqrt(p[0] ** 2 + p[1] ** 2)
        return (rho_xy <= r_rod) and (p[2] <= 0.0)

    geom = rod_openings.get(owner, None)

    if geom is None:
        # No rod-mesh samples found near this shell -> rod doesn't cross
        # it (or the sampling density was too low there) -> no opening.
        return False

    x0, y0 = geom["center_xy"]
    r_eff = geom["radius"]
    z_ref = geom["z_ref"]

    rho = np.sqrt((p[0] - x0) ** 2 + (p[1] - y0) ** 2)

    # A given (x, y) maps to two possible points on the sphere (+/- z);
    # only the side where the rod actually crosses should count as open.
    same_side = (p[2] <= 0.0) if z_ref <= 0.0 else (p[2] >= 0.0)

    return (rho <= r_eff) and same_side


def is_in_spherical_cap_opening(
    p,
    u_axis,
    alpha_deg: float,
) -> bool:
    """
    Check whether point is inside a spherical cap opening.

    The point is interpreted relative to the origin.
    """
    p = np.asarray(p, dtype=float)

    r = np.linalg.norm(p)

    if r == 0:
        return False

    u = np.asarray(u_axis, dtype=float)
    u = u / np.linalg.norm(u)

    cosang = np.dot(p, u) / r

    return cosang >= np.cos(np.deg2rad(alpha_deg))


def sphere_crossing_is_opening(
    p,
    owner: str,
    field: dict,
) -> bool:
    """
    Return True if a spherical-surface crossing is inside a known opening.

    The drift-tube and support-rod apertures pass through every shell
    (grids and collector alike), so they are checked unconditionally.
    Each grid additionally has its own azimuthal service/exchange cap.
    The collector explicitly has BOTH: the drift-tube aperture (checked
    below, shared with the grids) AND its own azimuthal service cap
    (COLLECTOR_OPENING_ALPHA_DEG) - matching the physical instrument,
    which has a beam-line exit as well as a separate access port.
    """
    p = np.asarray(p, dtype=float)

    if is_in_drift_tube_aperture(p, field):
        return True

    if is_in_rod_opening(p, owner, field):
        return True

    # Side service/opening cap direction, matching notebook convention.
    u_open = np.array([
        np.cos(np.deg2rad(225.0)),
        np.sin(np.deg2rad(225.0)),
        0.0,
    ])

    if owner == "g1_shell":
        return is_in_spherical_cap_opening(p, u_open, alpha_deg=20.0)

    if owner == "g2_shell":
        return is_in_spherical_cap_opening(p, u_open, alpha_deg=18.0)

    if owner == "g3_shell":
        return is_in_spherical_cap_opening(p, u_open, alpha_deg=14.0)

    if owner == "collector_shell":
        # Drift-tube aperture already checked above (shared with grids).
        # This adds the collector's own azimuthal service-port opening.
        return is_in_spherical_cap_opening(
            p,
            u_open,
            alpha_deg=COLLECTOR_OPENING_ALPHA_DEG,
        )

    return False


def first_analytic_grid_hit(
    p0,
    p1,
    field: dict,
    ignore_owners=None,
):
    """
    First analytic spherical-grid or collector-shell hit along segment.

    Openings are ignored, i.e. a crossing through an opening is not
    considered a hit.
    """
    if ignore_owners is None:
        ignore_owners = set()

    candidates = []

    for name, radius in [
        ("g1_shell", field["R_g1"]),
        ("g2_shell", field["R_g2"]),
        ("g3_shell", field["R_g3"]),
        ("collector_shell", field["R_col"]),
    ]:
        if name in ignore_owners:
            continue

        hit = first_sphere_segment_crossing(
            p0=p0,
            p1=p1,
            radius=radius,
            name=name,
        )

        if hit is None:
            continue

        if sphere_crossing_is_opening(hit["location"], name, field):
            continue

        candidates.append(hit)

    if not candidates:
        return None

    return min(candidates, key=lambda h: h["distance"])


def classify_sphere_event(hit: dict) -> str:
    """
    Convert analytic sphere hit to high-level event type.
    """
    owner = hit["owner"]

    if owner in ["g1_shell", "g2_shell", "g3_shell"]:
        return "transmit_grid"

    if owner == "collector_shell":
        return "hit_collector"

    return "hit_sphere"


def nearest_hit(*hits):
    """
    Return nearest hit from a list of possible hit dictionaries.
    """
    hits = [h for h in hits if h is not None]

    if not hits:
        return None

    return min(hits, key=lambda h: h["distance"])


# ============================================================
# Analytic sample plane
# ============================================================

def make_sample_plane_geometry(
    sample_y_bounds,
    sample_z_bounds,
    theta_deg: float = 0.0,
    x_sample: float = 0.0,
):
    """Build the finite rotated sample plane used by launch and tracking.

    The unrotated sample front face is the plane ``x=x_sample`` with its
    vacuum-side normal along +X.  ``theta_deg`` rotates the sample around the
    global +Z axis through the centre of the sample face.  The original
    y/z bounds therefore become bounds in the sample's local in-plane
    coordinates (u, v), not axis-aligned global bounds after rotation.

    This representation is continuous and independent of the electrostatic
    voxel spacing.  The voxel grid is still used to solve the field and to
    verify that a launch point is in vacuum, but it no longer defines where a
    finite beam ray intersects the physical sample surface.
    """
    if sample_y_bounds is None or sample_z_bounds is None:
        raise ValueError(
            "sample_y_bounds and sample_z_bounds are required to build "
            "rotated sample geometry"
        )

    y_lo, y_hi = map(float, sample_y_bounds)
    z_lo, z_hi = map(float, sample_z_bounds)

    y_c = 0.5 * (y_lo + y_hi)
    z_c = 0.5 * (z_lo + z_hi)

    a = np.deg2rad(float(theta_deg))
    ca = float(np.cos(a))
    sa = float(np.sin(a))

    normal = np.array([ca, sa, 0.0], dtype=float)
    u_axis = np.array([-sa, ca, 0.0], dtype=float)
    v_axis = np.array([0.0, 0.0, 1.0], dtype=float)

    return {
        "center": np.array([float(x_sample), y_c, z_c], dtype=float),
        "normal": normal,
        "u_axis": u_axis,
        "v_axis": v_axis,
        "u_bounds": (y_lo - y_c, y_hi - y_c),
        "v_bounds": (z_lo - z_c, z_hi - z_c),
        "theta_deg": float(theta_deg),
        "x_sample_unrotated": float(x_sample),
    }


def _validated_sample_geometry(sample_geometry):
    """Return a normalized copy of a sample-geometry dictionary."""
    if not isinstance(sample_geometry, dict):
        raise TypeError("sample_geometry must be a dict")

    center = np.asarray(sample_geometry["center"], dtype=float)
    normal = np.asarray(sample_geometry["normal"], dtype=float)
    u_axis = np.asarray(sample_geometry["u_axis"], dtype=float)
    v_axis = np.asarray(sample_geometry["v_axis"], dtype=float)

    if not all(_is_finite_vec(v) for v in (center, normal, u_axis, v_axis)):
        raise ValueError("sample_geometry contains a non-finite 3-vector")

    def _norm(v, name):
        mag = float(np.linalg.norm(v))
        if not np.isfinite(mag) or mag <= 0.0:
            raise ValueError(f"sample_geometry[{name!r}] has zero/invalid norm")
        return v / mag

    normal = _norm(normal, "normal")
    u_axis = _norm(u_axis, "u_axis")
    v_axis = _norm(v_axis, "v_axis")

    out = dict(sample_geometry)
    out.update({
        "center": center,
        "normal": normal,
        "u_axis": u_axis,
        "v_axis": v_axis,
    })
    return out


def sample_plane_coordinates(p, sample_geometry):
    """Return signed normal distance and local (u, v) coordinates."""
    geom = _validated_sample_geometry(sample_geometry)
    p = np.asarray(p, dtype=float)
    rel = p - geom["center"]
    return (
        float(np.dot(rel, geom["normal"])),
        float(np.dot(rel, geom["u_axis"])),
        float(np.dot(rel, geom["v_axis"])),
    )


def sample_point_is_in_bounds(location, sample_geometry, tol: float = 1.0e-12):
    """True if a point on the sample plane lies inside the finite sample face."""
    geom = _validated_sample_geometry(sample_geometry)
    _, u, v = sample_plane_coordinates(location, geom)

    u_bounds = geom.get("u_bounds", None)
    v_bounds = geom.get("v_bounds", None)

    if u_bounds is not None:
        if u < float(u_bounds[0]) - tol or u > float(u_bounds[1]) + tol:
            return False
    if v_bounds is not None:
        if v < float(v_bounds[0]) - tol or v > float(v_bounds[1]) + tol:
            return False

    return True


def ray_hits_sample_plane(
    ray_origin,
    ray_direction,
    sample_geometry,
    require_in_bounds: bool = True,
    eps: float = 1.0e-15,
):
    """Intersect a forward ray with the finite oriented sample plane.

    ``ray_direction`` need not be normalized.  The returned ``distance`` is
    the physical distance from ``ray_origin`` to the intersection.
    """
    geom = _validated_sample_geometry(sample_geometry)
    p0 = np.asarray(ray_origin, dtype=float)
    d = np.asarray(ray_direction, dtype=float)

    if not _is_finite_vec(p0) or not _is_finite_vec(d):
        return None

    dnorm = float(np.linalg.norm(d))
    if not np.isfinite(dnorm) or dnorm <= eps:
        return None

    dhat = d / dnorm
    denom = float(np.dot(geom["normal"], dhat))
    if abs(denom) <= eps:
        return None

    distance_along_ray = float(
        np.dot(geom["normal"], geom["center"] - p0) / denom
    )
    if not np.isfinite(distance_along_ray) or distance_along_ray < -eps:
        return None

    distance_along_ray = max(0.0, distance_along_ray)
    location = p0 + distance_along_ray * dhat

    if require_in_bounds and not sample_point_is_in_bounds(location, geom):
        return None

    _, u, v = sample_plane_coordinates(location, geom)

    hit = {
        "kind": "sample_plane",
        "location": location,
        "distance": distance_along_ray,
        "owner": "sample",
        "normal": geom["normal"].copy(),
        "sample_u": u,
        "sample_v": v,
        "sample_theta_deg": geom.get("theta_deg", np.nan),
    }
    return add_owner_metadata(hit, owner_name="sample")


def segment_hits_sample_plane(
    p0,
    p1,
    x_sample: float = 0.0,
    sample_y_bounds=None,
    sample_z_bounds=None,
    sample_geometry=None,
):
    """Check whether segment ``p0 -> p1`` crosses the finite sample face.

    Preferred path
    --------------
    Pass ``sample_geometry`` from :func:`make_sample_plane_geometry`.  This
    supports a sample rotated around Z and checks the finite face in local
    in-plane coordinates.

    Backward-compatible path
    ------------------------
    If no geometry is supplied, the old ``x=x_sample`` plane is reconstructed
    from ``sample_y_bounds`` / ``sample_z_bounds`` with ``theta_deg=0``.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)

    if not _is_finite_vec(p0) or not _is_finite_vec(p1):
        return None

    if sample_geometry is None:
        if sample_y_bounds is None or sample_z_bounds is None:
            # Preserve the legacy unbounded x-plane behavior when bounds were
            # intentionally omitted.
            y_bounds = (-np.inf, np.inf)
            z_bounds = (-np.inf, np.inf)
        else:
            y_bounds = sample_y_bounds
            z_bounds = sample_z_bounds

        sample_geometry = make_sample_plane_geometry(
            y_bounds,
            z_bounds,
            theta_deg=0.0,
            x_sample=x_sample,
        )

    geom = _validated_sample_geometry(sample_geometry)
    n = geom["normal"]
    C = geom["center"]

    d0 = float(np.dot(p0 - C, n))
    d1 = float(np.dot(p1 - C, n))

    # Strictly same side -> no crossing.  A point exactly on the plane is
    # allowed so a segment that lands numerically on the surface is retained.
    if d0 * d1 > 0.0:
        return None

    denom = d0 - d1
    if abs(denom) < 1.0e-30:
        return None

    t = d0 / denom
    if not (0.0 <= t <= 1.0):
        return None

    location = p0 + t * (p1 - p0)
    if not sample_point_is_in_bounds(location, geom):
        return None

    _, u, v = sample_plane_coordinates(location, geom)

    hit = {
        "kind": "sample_plane",
        "location": location,
        "distance": float(np.linalg.norm(location - p0)),
        "t": float(t),
        "owner": "sample",
        "normal": n.copy(),
        "sample_u": u,
        "sample_v": v,
        "sample_theta_deg": geom.get("theta_deg", np.nan),
    }

    return add_owner_metadata(hit, owner_name="sample")


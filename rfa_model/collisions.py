"""
collisions.py

STL collision helpers, analytic spherical grid/collector crossings,
grid/opening logic, and analytic sample-plane intersections.
"""

from __future__ import annotations

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


def build_stl_intersector(meshes_for_collision: dict[str, trimesh.Trimesh]):
    """
    Build a trimesh ray intersector for a labeled collision mesh.
    """
    collision_mesh, face_owner = combine_labeled_meshes(meshes_for_collision)

    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(
        collision_mesh
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


def is_in_drift_tube_aperture(
    p,
    field: dict,
    r_hole: float | None = None,
) -> bool:
    """
    Check whether point on a spherical surface lies in the drift-tube aperture.

    The drift-tube aperture is near +X and has radius r_hole in the YZ plane.

    r_hole defaults to field["drifttube_aperture_radius"] when present, and
    falls back to 5.6 mm (0.0056 m) otherwise. This is the single source of
    truth for the drift-tube bore radius, also used by
    trajectories.is_drifttube_escape_candidate() for the domain-boundary
    escape check, so both checks agree on the same physical aperture.

    NOTE: this is still the fixed-circle approximation (unlike
    is_in_rod_opening, which is now derived from the real rod STL). If the
    drift tube also turns out to be measurably off-axis, the same
    compute_rod_opening_geometry() approach can be reused here.
    """
    p = np.asarray(p, dtype=float)

    if r_hole is None:
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

def segment_hits_sample_plane(
    p0,
    p1,
    x_sample: float = 0.0,
    sample_y_bounds=None,
    sample_z_bounds=None,
):
    """
    Check if segment p0 -> p1 crosses the analytic sample plane x=x_sample.

    Optionally require the crossing point to lie within sample y/z bounds.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)

    x0 = p0[0] - x_sample
    x1 = p1[0] - x_sample

    if x0 * x1 > 0:
        return None

    if abs(x0 - x1) < 1e-30:
        return None

    t = x0 / (x0 - x1)

    if not (0.0 <= t <= 1.0):
        return None

    location = p0 + t * (p1 - p0)

    if sample_y_bounds is not None:
        if not (sample_y_bounds[0] <= location[1] <= sample_y_bounds[1]):
            return None

    if sample_z_bounds is not None:
        if not (sample_z_bounds[0] <= location[2] <= sample_z_bounds[1]):
            return None

    hit = {
        "kind": "sample_plane",
        "location": location,
        "distance": np.linalg.norm(location - p0),
        "owner": "sample",
        "normal": np.array([1.0, 0.0, 0.0]),
    }

    return add_owner_metadata(hit, owner_name="sample")
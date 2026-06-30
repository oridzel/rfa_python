"""
collisions.py

STL collision helpers, analytic spherical grid/collector crossings,
grid/opening logic, and analytic sample-plane intersections.
"""

from __future__ import annotations

import numpy as np
import trimesh


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

    seg = p1 - p0
    seg_len = np.linalg.norm(seg)

    if seg_len <= eps:
        return None

    direction = seg / seg_len

    locations, ray_ids, tri_ids = intersector.intersects_location(
        ray_origins=p0.reshape(1, 3),
        ray_directions=direction.reshape(1, 3),
        multiple_hits=False,
    )

    if len(locations) == 0:
        return None

    location = locations[0]
    tri_id = int(tri_ids[0])

    distance = np.linalg.norm(location - p0)

    if distance <= min_distance:
        return None

    if distance > seg_len + eps:
        return None

    owner = face_owner[tri_id]
    normal = collision_mesh.face_normals[tri_id]

    return {
        "kind": "stl",
        "location": location,
        "distance": distance,
        "face_id": tri_id,
        "owner": owner,
        "normal": normal,
    }


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

    d = p1 - p0

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

    Returns None if there is no crossing inside the segment.
    """
    p0_shift = np.asarray(p0, dtype=float) - center
    p1_shift = np.asarray(p1, dtype=float) - center

    d = p1_shift - p0_shift

    a = np.dot(d, d)
    b = 2.0 * np.dot(p0_shift, d)
    c = np.dot(p0_shift, p0_shift) - radius**2

    disc = b * b - 4.0 * a * c

    if disc < 0 or a <= eps:
        return None

    sqrt_disc = np.sqrt(disc)

    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    candidates = [t for t in (t1, t2) if eps < t <= 1.0 + eps]

    if not candidates:
        return None

    t = min(candidates)

    location = (p0_shift + t * d) + center

    normal = location - center
    normal = normal / np.linalg.norm(normal)

    return {
        "kind": "sphere",
        "location": location,
        "distance": np.linalg.norm(t * d),
        "t": t,
        "owner": name,
        "normal": normal,
    }


# ============================================================
# Grid / collector openings
# ============================================================

def is_in_drift_tube_aperture(
    p,
    field: dict,
    r_hole: float = 0.0056,
) -> bool:
    """
    Check whether point on a spherical surface lies in the drift-tube aperture.

    The drift-tube aperture is near +X and has radius r_hole in the YZ plane.
    """
    p = np.asarray(p, dtype=float)

    rho_yz = np.sqrt(p[1]**2 + p[2]**2)

    return (p[0] > 0.0) and (rho_yz <= r_hole)


def is_in_rod_opening(
    p,
    field: dict,
    r_rod: float = 0.011,
) -> bool:
    """
    Check whether point is in the lower rod opening.
    """
    p = np.asarray(p, dtype=float)

    rho_xy = np.sqrt(p[0]**2 + p[1]**2)

    return (rho_xy <= r_rod) and (p[2] <= 0.0)


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
    """
    p = np.asarray(p, dtype=float)

    if is_in_drift_tube_aperture(p, field, r_hole=0.0056):
        return True

    if is_in_rod_opening(p, field, r_rod=0.011):
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
        return is_in_spherical_cap_opening(p, u_open, alpha_deg=14.0)

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

    return {
        "kind": "sample_plane",
        "location": location,
        "distance": np.linalg.norm(location - p0),
        "owner": "sample",
        "normal": np.array([1.0, 0.0, 0.0]),
    }
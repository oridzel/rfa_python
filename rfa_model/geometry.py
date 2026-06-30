"""
geometry.py

STL loading, rigid transformations, RFA alignment, and grid-frame alignment
for the RFA trajectory model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


# ============================================================
# Rotation helpers
# ============================================================

def Rx(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(a), -np.sin(a)],
        [0.0, np.sin(a),  np.cos(a)],
    ])


def Ry(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([
        [ np.cos(a), 0.0, np.sin(a)],
        [0.0,        1.0, 0.0],
        [-np.sin(a), 0.0, np.cos(a)],
    ])


def Rz(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([
        [np.cos(a), -np.sin(a), 0.0],
        [np.sin(a),  np.cos(a), 0.0],
        [0.0,        0.0,       1.0],
    ])


# ============================================================
# STL part loader
# ============================================================

@dataclass
class Part:
    name: str
    filename: str
    rotation: np.ndarray | None = None
    translation: np.ndarray | None = None
    center: bool = False
    scale: float = 1e-3
    voltage: float | None = None
    color: str = "lightgray"

    def load(self, stl_dir: str | Path) -> trimesh.Trimesh:
        """
        Load STL, apply scale, optional centering, rotation, translation.

        Parameters
        ----------
        stl_dir:
            Directory containing STL file.

        Returns
        -------
        trimesh.Trimesh
        """
        stl_dir = Path(stl_dir)
        mesh = trimesh.load(stl_dir / self.filename)

        V = mesh.vertices.copy() * self.scale

        if self.center:
            V -= V.mean(axis=0)

        if self.rotation is not None:
            V = (self.rotation @ V.T).T

        if self.translation is not None:
            V += np.asarray(self.translation, dtype=float)

        return trimesh.Trimesh(vertices=V, faces=mesh.faces, process=False)


def load_parts(parts: list[Part], stl_dir: str | Path) -> dict[str, trimesh.Trimesh]:
    """
    Load a list of Part objects into a dictionary of meshes.
    """
    meshes = {}
    for part in parts:
        meshes[part.name] = part.load(stl_dir)
    return meshes


# ============================================================
# Main sample / holder / receiver / rod alignment
# ============================================================

def _lower_z_mask(V, dz=0.010):
    """
    Vertices within dz of the minimum z.
    """
    zmin = V[:, 2].min()
    mask = V[:, 2] <= zmin + dz

    if not np.any(mask):
        z_cut = np.percentile(V[:, 2], 20)
        mask = V[:, 2] <= z_cut

    return mask


def _upper_z_mask(V, dz=0.010):
    """
    Vertices within dz of the maximum z.
    """
    zmax = V[:, 2].max()
    mask = V[:, 2] >= zmax - dz

    if not np.any(mask):
        z_cut = np.percentile(V[:, 2], 80)
        mask = V[:, 2] >= z_cut

    return mask


def _middle_z_mask(V, z_fraction_low=0.45, z_fraction_high=0.55):
    """
    Vertices in a central z-slice, using relative z extent.
    """
    zmin = V[:, 2].min()
    zmax = V[:, 2].max()
    zlo = zmin + z_fraction_low * (zmax - zmin)
    zhi = zmin + z_fraction_high * (zmax - zmin)

    mask = (V[:, 2] >= zlo) & (V[:, 2] <= zhi)

    if not np.any(mask):
        z_cut = np.median(V[:, 2])
        dz = 0.1 * (zmax - zmin)
        mask = np.abs(V[:, 2] - z_cut) <= dz

    return mask


def default_sample_parts() -> list[Part]:
    """
    Default sample-holder-receiver-rod STL parts and initial rotations.
    """
    return [
        Part(
            name="sample",
            filename="sample gh.stl",
            rotation=Rx(180) @ Ry(90),
            center = True,
            voltage=0.0,
            color="gold",
        ),
        Part(
            name="holder",
            filename="Sample holder WO JS.stl",
            rotation=Rx(180) @ Ry(90),
            center = True,
            voltage=0.0,
            color="lightgray",
        ),
        Part(
            name="receiver",
            filename="Receiver only.stl",
            rotation=Ry(90) @ Rz(180),
            center = True,
            voltage=0.0,
            color="lightblue",
        ),
        Part(
            name="rod",
            filename="Support rod for sample.stl",
            rotation=Rx(180) @ Rz(90),
            center = True,
            voltage=0.0,
            color="lightgreen",
        ),
    ]


def apply_rfa_alignment(meshes, alpha_deg=0.0):
    """
    RFA alignment block adapted from the MATLAB alignment logic.

    This version avoids hard-coded absolute z thresholds because the current
    STL files have different z origins than the original MATLAB test files.
    """

    V_Rod = meshes["rod"].vertices.copy()
    F_Rod = meshes["rod"].faces.copy()

    V_Receiver = meshes["receiver"].vertices.copy()
    F_Receiver = meshes["receiver"].faces.copy()

    V_Holder = meshes["holder"].vertices.copy()
    F_Holder = meshes["holder"].faces.copy()

    V_Sample = meshes["sample"].vertices.copy()
    F_Sample = meshes["sample"].faces.copy()

    # ------------------------------------------------------------
    # 1. Center rod lower section in X and Y
    # ------------------------------------------------------------
    ind_rod_bottom = _lower_z_mask(V_Rod, dz=0.010)

    V_Rod[:, 0] -= (
        V_Rod[ind_rod_bottom, 0].max()
        + V_Rod[ind_rod_bottom, 0].min()
    ) / 2.0

    V_Rod[:, 1] -= (
        V_Rod[ind_rod_bottom, 1].max()
        + V_Rod[ind_rod_bottom, 1].min()
    ) / 2.0

    # ------------------------------------------------------------
    # 2. Align receiver Y to rod upper section
    # ------------------------------------------------------------
    ind_rod_upper_raw = V_Rod[:, 2] > 0.01

    if not np.any(ind_rod_upper_raw):
        ind_rod_upper_raw = _upper_z_mask(V_Rod, dz=0.010)

    shift_rec_y = V_Rod[ind_rod_upper_raw, 1].min() - V_Receiver[:, 1].min()
    V_Receiver[:, 1] += shift_rec_y

    # ------------------------------------------------------------
    # 3. Align rod Z to receiver Z
    # ------------------------------------------------------------
    shift_Z = V_Rod[:, 2].max() - V_Receiver[:, 2].max()
    V_Rod[:, 2] -= shift_Z

    # ------------------------------------------------------------
    # 4. Align receiver X to rod upper section, with 0.00127 m spacing
    # ------------------------------------------------------------
    ind_rod_upper = V_Rod[:, 2] > 0.0

    if not np.any(ind_rod_upper):
        ind_rod_upper = _upper_z_mask(V_Rod, dz=0.010)

    shift_rec_x = V_Rod[ind_rod_upper, 0].max() + 0.00127 - V_Receiver[:, 0].min()
    V_Receiver[:, 0] += shift_rec_x

    # ------------------------------------------------------------
    # 5. Align holder Y to receiver upper section
    # ------------------------------------------------------------
    ind_receiver_upper = V_Receiver[:, 2] > 0.005

    if not np.any(ind_receiver_upper):
        ind_receiver_upper = _upper_z_mask(V_Receiver, dz=0.002)

    shift_Y = V_Holder[:, 1].max() - V_Receiver[ind_receiver_upper, 1].max()
    V_Holder[:, 1] -= shift_Y

    # ------------------------------------------------------------
    # 6. Align holder X
    # Original code used a particular absolute z slice.
    # Current holder z is only about -2.75 to 0.75 mm, so use a robust
    # side feature selection.
    # ------------------------------------------------------------
    ind_z = (V_Holder[:, 2] > 0.003) & (V_Holder[:, 2] < 0.004)
    ind_x = V_Holder[:, 0] > 0.0
    ind_y = V_Holder[:, 1] < -0.007

    mask_holder_x = ind_x & ind_y & ind_z

    if not np.any(mask_holder_x):
        # Current STL fallback: use same x/y feature without impossible z slice.
        mask_holder_x = ind_x & ind_y

    if not np.any(mask_holder_x):
        # Final fallback: use positive-x side of holder.
        mask_holder_x = ind_x

    shift_x = V_Holder[mask_holder_x, 0].min()
    V_Holder[:, 0] -= shift_x

    # ------------------------------------------------------------
    # 7. Move sample front face to X = 0
    # ------------------------------------------------------------
    V_Sample[:, 0] -= V_Sample[:, 0].max()

    # ------------------------------------------------------------
    # 8. Center sample in Y using holder front face
    # ------------------------------------------------------------
    front_X = V_Holder[:, 0].max()
    tolerance = 1e-4

    idx_front = np.abs(V_Holder[:, 0] - front_X) < tolerance

    if not np.any(idx_front):
        idx_front = V_Holder[:, 0] >= front_X - 0.001

    V_front = V_Holder[idx_front, :]

    center_Y = (V_front[:, 1].max() + V_front[:, 1].min()) / 2.0
    global_shift = np.array([0.0, center_Y, 0.0])

    V_Sample += global_shift

    # ------------------------------------------------------------
    # 9. Optional rotation around Z
    # ------------------------------------------------------------
    if alpha_deg != 0.0:
        R_z = Rz(alpha_deg)

        V_Sample = (R_z @ V_Sample.T).T
        V_Receiver = (R_z @ V_Receiver.T).T
        V_Holder = (R_z @ V_Holder.T).T
        V_Rod = (R_z @ V_Rod.T).T

    aligned = dict(meshes)

    aligned["rod"] = trimesh.Trimesh(vertices=V_Rod, faces=F_Rod, process=False)
    aligned["receiver"] = trimesh.Trimesh(vertices=V_Receiver, faces=F_Receiver, process=False)
    aligned["holder"] = trimesh.Trimesh(vertices=V_Holder, faces=F_Holder, process=False)
    aligned["sample"] = trimesh.Trimesh(vertices=V_Sample, faces=F_Sample, process=False)

    return aligned


def load_and_align_sample_assembly(
    stl_dir: str | Path,
    alpha_deg: float = 0.0,
) -> tuple[dict[str, trimesh.Trimesh], list[Part]]:
    """
    Load and align the sample/holder/receiver/rod assembly using the
    tested default pre-rotations.

    Returns
    -------
    meshes:
        Dict with sample, holder, receiver, rod.
    sample_parts:
        The Part definitions used for loading.
    """
    sample_parts = default_sample_parts()
    meshes = load_parts(sample_parts, stl_dir)
    meshes = apply_rfa_alignment(meshes, alpha_deg=alpha_deg)

    return meshes, sample_parts


# ============================================================
# Grid-frame alignment helpers
# ============================================================

def fit_sphere_center(V: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Least-squares fit of sphere center and radius to vertices.

    Returns
    -------
    C:
        Sphere center.
    R:
        Sphere radius.
    residuals:
        Least-squares residuals.
    """
    V = np.asarray(V, dtype=float)

    x = V[:, 0]
    y = V[:, 1]
    z = V[:, 2]

    A = np.column_stack([2.0 * x, 2.0 * y, 2.0 * z, np.ones_like(x)])
    b = x**2 + y**2 + z**2

    sol, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    C = sol[:3]
    c = sol[3]
    R = np.sqrt(c + np.dot(C, C))

    return C, R, residuals


def frame_mid_radius(V: np.ndarray) -> tuple[float, float, float, float]:
    """
    Estimate grid-frame mid-radius from vertex radial distribution.

    Returns
    -------
    Rmid, thickness, Rin, Rout
    """
    V = np.asarray(V, dtype=float)
    r = np.linalg.norm(V, axis=1)

    Rin = np.percentile(r, 2)
    Rout = np.percentile(r, 98)
    Rmid = 0.5 * (Rin + Rout)
    thickness = Rout - Rin

    return Rmid, thickness, Rin, Rout


def rotvec_to_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Rotation matrix that maps vector a to vector b.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    v = np.cross(a, b)
    c = np.dot(a, b)

    if np.isclose(c, 1.0):
        return np.eye(3)

    if np.isclose(c, -1.0):
        tmp = np.array([1.0, 0.0, 0.0])
        if np.allclose(np.abs(a), tmp):
            tmp = np.array([0.0, 1.0, 0.0])

        axis = np.cross(a, tmp)
        axis /= np.linalg.norm(axis)

        K = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ])

        return np.eye(3) + 2.0 * K @ K

    s = np.linalg.norm(v)

    K = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])

    return np.eye(3) + K + K @ K * ((1.0 - c) / s**2)


def default_grid_frame_parts() -> list[Part]:
    """
    Default grid-frame STL parts and initial rotations.
    """
    return [
        Part(
            name="g1_low_frame",
            filename="3D printed Lower 3_5 inch.stl",
            rotation=Rz(135.0) @ Rx(180.0),
            color="orange",
        ),
        Part(
            name="g1_upper_frame",
            filename="3d printed upper 3_5 inch.stl",
            rotation=Rz(-45.0),
            color="orange",
        ),
        Part(
            name="g2_low_frame",
            filename="3D printed lower 4_5 frame.stl",
            rotation=Rz(135.0) @ Rx(180.0),
            color="green",
        ),
        Part(
            name="g2_upper_frame",
            filename="3D printed upper 4_5 frame.stl",
            rotation=Rz(-45.0),
            color="green",
        ),
        Part(
            name="g3_low_frame",
            filename="3D printed Lower 5_5 inch frame.stl",
            rotation=Rz(135.0) @ Rx(180.0),
            color="purple",
        ),
        Part(
            name="g3_upper_frame",
            filename="3D printed UPPER 5_5 inch.stl",
            rotation=Rz(-45.0),
            color="purple",
        ),
    ]


def load_and_align_grid_frames(
    stl_dir: str | Path,
    verbose: bool = True,
) -> tuple[dict[str, trimesh.Trimesh], list[Part], dict]:
    """
    Load and align grid-frame STLs.

    Returns
    -------
    frame_meshes:
        Dict of aligned grid-frame meshes.
    frame_parts:
        Corresponding Part list.
    frame_info:
        Dict containing R_g1, R_g2, R_g3 and alignment diagnostics.
    """
    frame_parts = default_grid_frame_parts()
    frame_meshes = load_parts(frame_parts, stl_dir)

    # Combine all frame vertices for common sphere-center fit.
    all_vertices = np.vstack([m.vertices for m in frame_meshes.values()])

    Cfit, Rfit, residuals = fit_sphere_center(all_vertices)

    # First center all frames on fitted sphere center.
    centered_vertices = {}
    for name, mesh in frame_meshes.items():
        centered_vertices[name] = mesh.vertices.copy() - Cfit

    # Estimate grid radii from grouped frames.
    V_g1 = np.vstack([
        centered_vertices["g1_low_frame"],
        centered_vertices["g1_upper_frame"],
    ])
    V_g2 = np.vstack([
        centered_vertices["g2_low_frame"],
        centered_vertices["g2_upper_frame"],
    ])
    V_g3 = np.vstack([
        centered_vertices["g3_low_frame"],
        centered_vertices["g3_upper_frame"],
    ])

    R_g1, t_g1, Rin_g1, Rout_g1 = frame_mid_radius(V_g1)
    R_g2, t_g2, Rin_g2, Rout_g2 = frame_mid_radius(V_g2)
    R_g3, t_g3, Rin_g3, Rout_g3 = frame_mid_radius(V_g3)

    # Find approximate drift-tube hole centers for each frame group.
    def hole_center(V: np.ndarray, Rg: float) -> np.ndarray:
        rho_yz = np.sqrt(V[:, 1]**2 + V[:, 2]**2)
        r = np.linalg.norm(V, axis=1)

        mask = (
            (np.abs(r - Rg) < 1.0e-3)
            & (V[:, 0] > 0.0)
            & (rho_yz < 0.01)
        )

        if not np.any(mask):
            # Fallback: use high-X vertices near axis.
            mask = (V[:, 0] > np.percentile(V[:, 0], 95)) & (rho_yz < 0.02)

        if not np.any(mask):
            raise RuntimeError("Could not find grid-frame drift-tube hole center.")

        return V[mask].mean(axis=0)

    c1 = hole_center(V_g1, R_g1)
    c2 = hole_center(V_g2, R_g2)
    c3 = hole_center(V_g3, R_g3)

    centers = np.vstack([c1, c2, c3])
    Cmean = centers.mean(axis=0)

    # Fit axis through hole centers.
    _, _, Vt = np.linalg.svd(centers - Cmean)
    axis_frame = Vt[0, :]

    if axis_frame[0] < 0.0:
        axis_frame = -axis_frame

    # Rotate fitted axis to +X.
    Rax = rotvec_to_vec(axis_frame, np.array([1.0, 0.0, 0.0]))

    rotated_vertices = {}
    for name, V in centered_vertices.items():
        rotated_vertices[name] = (Rax @ V.T).T

    # Remove residual Y/Z offset of hole-axis mean.
    C_rot = (Rax @ Cmean.reshape(3, 1)).ravel()
    shift2 = np.array([0.0, C_rot[1], C_rot[2]])

    aligned_meshes = {}
    for part in frame_parts:
        name = part.name
        V = rotated_vertices[name] - shift2
        F = frame_meshes[name].faces.copy()

        aligned_meshes[name] = trimesh.Trimesh(
            vertices=V,
            faces=F,
            process=False,
        )

    frame_info = {
        "Cfit": Cfit,
        "Rfit": Rfit,
        "residuals": residuals,
        "R_g1": R_g1,
        "R_g2": R_g2,
        "R_g3": R_g3,
        "thickness_g1": t_g1,
        "thickness_g2": t_g2,
        "thickness_g3": t_g3,
        "Rin_g1": Rin_g1,
        "Rout_g1": Rout_g1,
        "Rin_g2": Rin_g2,
        "Rout_g2": Rout_g2,
        "Rin_g3": Rin_g3,
        "Rout_g3": Rout_g3,
        "hole_centers_before_axis_alignment": centers,
        "axis_frame_before_alignment": axis_frame,
        "axis_rotation": Rax,
        "axis_shift_yz": shift2,
    }

    if verbose:
        print("Grid frame alignment")
        print("  fitted center:", Cfit)
        print("  fitted radius:", Rfit)
        print("  axis before alignment:", axis_frame)
        print("  R_g1:", R_g1)
        print("  R_g2:", R_g2)
        print("  R_g3:", R_g3)

    return aligned_meshes, frame_parts, frame_info


# ============================================================
# Convenience builders
# ============================================================

def build_collision_mesh_dict(
    meshes: dict[str, trimesh.Trimesh],
    frame_meshes: dict[str, trimesh.Trimesh],
    include_sample: bool = True,
) -> dict[str, trimesh.Trimesh]:
    """
    Build a collision-mesh dictionary.

    include_sample=True:
        for primary beam or general STL collision.

    include_sample=False:
        for emitted electron tracking where sample return is handled
        analytically by sample plane.
    """
    collision_meshes = {}

    if include_sample:
        collision_meshes["sample"] = meshes["sample"]

    collision_meshes.update({
        "holder": meshes["holder"],
        "receiver": meshes["receiver"],
        "rod": meshes["rod"],

        "g1_low_frame": frame_meshes["g1_low_frame"],
        "g1_upper_frame": frame_meshes["g1_upper_frame"],
        "g2_low_frame": frame_meshes["g2_low_frame"],
        "g2_upper_frame": frame_meshes["g2_upper_frame"],
        "g3_low_frame": frame_meshes["g3_low_frame"],
        "g3_upper_frame": frame_meshes["g3_upper_frame"],
    })

    return collision_meshes


def sample_bounds(meshes: dict[str, trimesh.Trimesh]) -> tuple[np.ndarray, np.ndarray]:
    """
    Return sample y and z bounds.

    Useful for analytic sample-plane primary hit / sample return.
    """
    bounds = meshes["sample"].bounds

    sample_y_bounds = bounds[:, 1].copy()
    sample_z_bounds = bounds[:, 2].copy()

    return sample_y_bounds, sample_z_bounds
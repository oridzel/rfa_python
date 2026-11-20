"""
Electron trajectories through / off one cell of a square wire mesh in a planar
local model, with the cell-scale field computed analytically.

Mesh: two orthogonal arrays of parallel wires in the plane z = 0, pitch p,
wire radius a.  Each array is represented by line charges; the potential is
    phi = V0 - (g/2) z + s * C * [ ln(2(cosh kz - cos kx)) + ln(2(cosh kz - cos ky)) ],
    k = 2 pi / p,  C = g p / (8 pi)
which has zero field on one side of the mesh and a uniform field g on the
other (s = +1: field on the z < 0 side, potential rising away from the mesh;
s = -1: field on the z > 0 side, potential falling away from the mesh).
The plane-average potential at z = 0 is V0.  Valid for a << p (here a/p = 0.025);
the wires are not exactly equipotential near the crossings in this model.
"""
import numpy as np

QM = 1.75882001e11          # e/m, C/kg
P = 0.508e-3                # pitch, m
A = 12.7e-6                 # wire radius, m
K = 2 * np.pi / P


def potential(x, y, z, g, s, c=None):
    C = g * P / (8 * np.pi)
    c = -g / 2 if c is None else c
    ch = np.cosh(K * z)
    return c * z + s * C * (np.log(2 * (ch - np.cos(K * x))) + np.log(2 * (ch - np.cos(K * y))))


def grad(x, y, z, g, s, c=None):
    C = g * P / (8 * np.pi)
    c = -g / 2 if c is None else c
    ch, sh = np.cosh(K * z), np.sinh(K * z)
    dx = ch - np.cos(K * x)
    dy = ch - np.cos(K * y)
    gx = s * C * K * np.sin(K * x) / dx
    gy = s * C * K * np.sin(K * y) / dy
    gz = c + s * C * K * sh * (1 / dx + 1 / dy)
    return gx, gy, gz


def hits_wire(x, y, z):
    ux = (x + P / 2) % P - P / 2          # distance to nearest wire of array along y
    uy = (y + P / 2) % P - P / 2
    return (ux ** 2 + z ** 2 < A ** 2) | (uy ** 2 + z ** 2 < A ** 2)


def run(K_start_eV, g, s, z0, n=4000, seed=1, theta_deg=0.0, dt=2e-12, tmax=4e-7, c=None, zfar=None):
    """Launch n electrons at z = z0 < 0 moving +z with kinetic energy K_start_eV,
    entry points uniform over the cell, polar angle theta_deg (azimuth random).
    Returns dict of arrays: outcome (0 absorbed, 1 crossed to z > +zfar, 2 returned
    to z < z0), final velocity."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, P, n); y = rng.uniform(0, P, n); z = np.full(n, z0)
    v0 = np.sqrt(2 * QM * K_start_eV)
    th = np.deg2rad(theta_deg); ph = rng.uniform(0, 2 * np.pi, n)
    vx = v0 * np.sin(th) * np.cos(ph); vy = v0 * np.sin(th) * np.sin(ph); vz = np.full(n, v0 * np.cos(th))
    out = np.full(n, -1)
    zfar = abs(z0) if zfar is None else zfar
    t = 0.0
    alive = np.ones(n, bool)
    while alive.any() and t < tmax:
        i = alive
        # velocity Verlet
        gx, gy, gz = grad(x[i], y[i], z[i], g, s, c)
        ax, ay, az = QM * gx, QM * gy, QM * gz
        # adaptive step: limit displacement to 2 um and velocity change
        sp = np.sqrt(vx[i] ** 2 + vy[i] ** 2 + vz[i] ** 2) + 1e3
        acc = np.sqrt(ax ** 2 + ay ** 2 + az ** 2) + 1.0
        h = np.minimum(np.minimum(2e-6 / sp, np.sqrt(4e-6 / acc)), dt * 50)
        xn = x[i] + vx[i] * h + 0.5 * ax * h * h
        yn = y[i] + vy[i] * h + 0.5 * ay * h * h
        zn = z[i] + vz[i] * h + 0.5 * az * h * h
        gx2, gy2, gz2 = grad(xn, yn, zn, g, s, c)
        vx[i] += 0.5 * (ax + QM * gx2) * h
        vy[i] += 0.5 * (ay + QM * gy2) * h
        vz[i] += 0.5 * (az + QM * gz2) * h
        x[i], y[i], z[i] = xn, yn, zn
        t += float(np.median(h))
        idx = np.where(i)[0]
        hw = hits_wire(xn, yn, zn)
        out[idx[hw]] = 0
        out[idx[(~hw) & (zn > zfar)]] = 1
        out[idx[(~hw) & (zn < z0) & (vz[i] < 0)]] = 2
        alive = out < 0
    return dict(outcome=out, vx=vx, vy=vy, vz=vz)


def sag(g):
    """Potential at the opening centre minus potential at a wire surface midway
    between crossings, and minus potential at a crossing (both relative to V0 model)."""
    c = potential(P / 2, P / 2, 0.0, g, +1)
    w = potential(A, P / 2, 0.0, g, +1)
    xg = potential(A * np.cos(np.pi / 4), A * np.sin(np.pi / 4), 0.0, g, +1)
    return c - w, c - xg

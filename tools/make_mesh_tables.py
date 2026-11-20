"""
Mesh transfer-function tables vs kappa = K / (e * dE * p), from the one-cell model.

K   kinetic energy of the electron on the field-free side of the mesh (= its
    energy at the shell crossing in the smooth-field digital twin)
dE  magnitude of the field difference across the mesh (V/m)
p   mesh pitch

Mesh types (by the sign of the mesh surface charge, i.e. of E_r,out - E_r,in):
  "L": the field side lies at LOWER potential than the mesh (wires above the
       openings), e.g. G1 in retarding-grid mode.  An electron entering the field
       side is decelerated; for small kappa it turns back inside the mesh field.
  "H": the field side lies at HIGHER potential (wires below the openings), e.g.
       G2 in retarding-grid mode seen from the cavity side, G3 toward the collector.

Tables (dimensionless, scale-invariant):
  L_round_trip : electron from the field-free side, turns within 0.8 p, comes back.
                 survival (both crossings), quantiles of E_t / (e dE p) of the returned electrons
  L_single     : electron from the field-free side passing into the field side
                 (time reversal gives the same for the reverse crossing).
  H_single     : electron from the field-free side passing into the field side.
Each single table gives survival and quantiles of E_t/(e dE p) at 0.8 p beyond the mesh,
where the cell-scale field has decayed to < 1 %.
"""
import json
import numpy as np
from mesh_cell import run, P, A, QM

G = 5000.0                 # any value: results scale with g*p
GP = G * P
ZS = 0.8 * P
QL = np.linspace(0, 1, 41)  # quantile levels
N = 6000


def transverse(r, sel):
    vt2 = r['vx'][sel] ** 2 + r['vy'][sel] ** 2
    return vt2 / (2 * QM) / GP          # E_t / (e g p)


def table(kappas, mesh_type, mode):
    out = []
    for kap in kappas:
        K = kap * GP
        if mesh_type == "L":
            r = run(K, G, -1, -ZS, n=N, seed=11, zfar=ZS if mode == "single" else K / G + ZS, tmax=5e-6)
        else:
            r = run(K, G, +1, -ZS, n=N, seed=12, c=+G / 2, zfar=ZS, tmax=5e-6)
        o = r['outcome']
        want = 2 if mode == "round_trip" else 1
        sel = o == want
        et = transverse(r, sel) if sel.any() else np.zeros(1)
        out.append(dict(kappa=float(kap), survival=float(np.mean(sel)),
                        other=float(np.mean((o != want) & (o != 0))),
                        Et_quantiles=[float(q) for q in np.quantile(et, QL)]))
        print(mesh_type, mode, f"kappa {kap:7.3f}  survival {np.mean(sel):.3f}  other {np.mean((o != want) & (o != 0)):.3f}  "
              f"Et median {np.median(et):.4f}  p90 {np.quantile(et, 0.9):.4f}")
    return out


if __name__ == "__main__":
    k_rt = np.geomspace(0.02, 0.8, 12)
    k_s = np.geomspace(0.8, 40.0, 12)
    k_h = np.geomspace(0.02, 40.0, 16)
    tabs = {
        "description": __doc__,
        "pitch_m": P, "wire_radius_m": A,
        "optical_opacity_single": float(1 - (1 - 2 * A / P) ** 2),
        "kappa_round_trip_max": 0.8,
        "quantile_levels": [float(q) for q in QL],
        "L_round_trip": table(k_rt, "L", "round_trip"),
        "L_single": table(k_s, "L", "single"),
        "H_single": table(k_h, "H", "single"),
    }
    with open("mesh_transfer_tables.json", "w") as f:
        json.dump(tabs, f, indent=1)
    print("written mesh_transfer_tables.json")

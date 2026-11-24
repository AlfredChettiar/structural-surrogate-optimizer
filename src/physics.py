"""Structural mechanics for a parametric tubular two-wheeler rear sub-frame.

Two fidelity levels are provided:

* ``frame_fe_solve`` - high-fidelity *ground truth*: a planar Euler-Bernoulli frame finite-element
  model of one sub-frame rail + diagonal strut (mirrored for the second rail), with
    - pipe-bend flexibility factor ``k`` and stress-intensification factor ``i`` (ASME B31.3 style),
    - gusseted-joint stiffening and weld-toe stress concentration,
    - linear static solve (rider + pillion/luggage load scaled by the dynamic load factor),
    - generalized eigen-solve for the first natural frequency (payload lumped masses),
    - Goodman mean-stress corrected Basquin fatigue life at the weld hot-spot.
* ``closed_form`` - low-fidelity hand calculation (cantilever beam formulas, no bend physics),
  calibrated at the design-space centre. Used as the physics prior of the Physics-Guided MLP.

Units: mm, N, MPa, tonne (for dynamics), seconds. Outputs: kg, MPa, mm, Hz, log10(cycles).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg

from src.config import FATIGUE_LOG10_CAP, FEATURES

# ----------------------------------------------------------------------------- constants
E_MPA = 205_000.0                # steel Young's modulus
RHO_KG_MM3 = 7.85e-6             # steel density [kg/mm^3]
RHO_T_MM3 = 7.85e-9              # steel density [tonne/mm^3] for dynamics
P_RIDER_N = 600.0                # static rider load carried by the sub-frame (both rails)
P_TAIL_N = 300.0                 # static pillion/luggage load at the tail (both rails)
M_SEAT_T = 0.003                 # seat/rider-side lumped mass per rail [t]
M_TAIL_T = 0.005                 # tail lamp/luggage-rack lumped mass per rail [t]
TORSION_ECC_MM = 50.0            # lateral eccentricity producing rail torsion
TORSION_SHARE = 0.15             # share of vertical load acting with that eccentricity
GUSSET_AREA_MM2 = 1250.0         # triangular gusset plate in-plane area
GUSSET_ZONE_MM = 35.0            # length of tube stiffened by a gusset
CROSS_MEMBER_LEN_MM = 2 * 240.0  # two cross tubes, same section as rails
BRACKETS_KG = 0.30               # seat locks, tabs, brackets (constant)
FATIGUE_KF = 0.55                # weld/surface fatigue-strength reduction factor
BASQUIN_B = -0.11                # Basquin exponent
ELEM_LEN_MM = 20.0               # target straight element length
ARC_SEGMENTS = 6                 # elements per bend

# rail corner points (before filleting) and strut definition
_L1, _LD, _LT, _ANG = 200.0, 150.0, 100.0, np.radians(25.0)   # straight, inclined, tail lengths
_RAIL_CORNERS = np.array([[0.0, 0.0], [_L1, 0.0], [_L1 + _LD * np.cos(_ANG), _LD * np.sin(_ANG)], [0.0, 0.0]])
_RAIL_CORNERS[3] = _RAIL_CORNERS[2] + np.array([_LT, 0.0])
_JOINT_X = 170.0                       # strut-to-rail gusseted joint (on first straight)
_STRUT_BASE = np.array([40.0, -170.0])  # lower strut mount on main frame
_RIDER_X = _L1 + 0.5 * _LD * np.cos(_ANG)
_TAIL_X = float(_RAIL_CORNERS[3, 0])
_STRAIGHT_LEN_MM = _L1 + _LD + _LT + float(np.hypot(*(np.array([_JOINT_X, 0.0]) - _STRUT_BASE)))


# ----------------------------------------------------------------------------- section helpers
def section_properties(od: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Area [mm^2], second moment [mm^4] and section modulus [mm^3] of a round tube."""
    od = np.asarray(od, dtype=float)
    t = np.asarray(t, dtype=float)
    idia = np.clip(od - 2.0 * t, 0.0, None)
    area = np.pi / 4.0 * (od**2 - idia**2)
    inertia = np.pi / 64.0 * (od**4 - idia**4)
    return area, inertia, 2.0 * inertia / od


def bend_factors(od, t, radius):
    """ASME B31.3-style pipe-bend flexibility factor k and in-plane SIF i (both >= 1)."""
    r_mean = (np.asarray(od) - np.asarray(t)) / 2.0
    h = np.asarray(t) * np.asarray(radius) / r_mean**2
    k = np.maximum(1.0, 1.65 / h)
    i = np.maximum(1.0, 0.9 / h ** (2.0 / 3.0))
    return k, i


def joint_scf(t, tg):
    """Weld-toe stress concentration at a gusseted tube joint; thicker gussets relieve it."""
    return 1.0 + 2.2 / (1.0 + (np.asarray(tg) / np.asarray(t)) ** 1.5)


def fatigue_log10_life(peak_vm, yield_mpa, dlf):
    """Goodman-corrected Basquin life at the hot-spot for the service load spectrum.

    Static stress = peak / DLF. The service cycle oscillates about static with amplitude
    (DLF - 1)/2 x static (a deterministic proxy for a PSD-derived damage-equivalent amplitude).
    """
    peak_vm = np.asarray(peak_vm, dtype=float)
    yield_mpa = np.asarray(yield_mpa, dtype=float)
    dlf = np.asarray(dlf, dtype=float)
    s_static = peak_vm / dlf
    s_amp = s_static * (dlf - 1.0) / 2.0
    s_ult = 1.25 * yield_mpa + 60.0
    denom = np.clip(1.0 - s_static / s_ult, 1e-6, None)
    s_eq = s_amp / denom
    sf_prime = FATIGUE_KF * (s_ult + 345.0)
    with np.errstate(divide="ignore"):
        log10_n = np.log10(0.5) + np.log10(np.clip(s_eq / sf_prime, 1e-12, None)) / BASQUIN_B
    return np.clip(log10_n, 0.0, FATIGUE_LOG10_CAP)


# ----------------------------------------------------------------------------- geometry
def _fillet(points: np.ndarray, radius: float):
    """Return a list of (polyline, kind) segments for a corner path with filleted corners."""
    segments = []
    cursor = points[0]
    for k in range(1, len(points) - 1):
        c = points[k]
        d1 = (c - points[k - 1]) / np.linalg.norm(c - points[k - 1])
        d2 = (points[k + 1] - c) / np.linalg.norm(points[k + 1] - c)
        theta = np.arccos(np.clip(d1 @ d2, -1.0, 1.0))
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        a = radius * np.tan(theta / 2.0)
        start, end = c - a * d1, c + a * d2
        normal = np.array([-d1[1], d1[0]]) * (1.0 if cross > 0 else -1.0)
        centre = start + radius * normal
        v0 = start - centre
        sign = 1.0 if cross > 0 else -1.0
        phis = np.linspace(0.0, theta, ARC_SEGMENTS + 1) * sign
        rot = np.stack([np.cos(phis), -np.sin(phis), np.sin(phis), np.cos(phis)], axis=1).reshape(-1, 2, 2)
        arc = centre + np.einsum("nij,j->ni", rot, v0)
        segments.append((np.vstack([cursor, start]), "straight"))
        segments.append((arc, "bend"))
        cursor = end
    segments.append((np.vstack([cursor, points[-1]]), "straight"))
    return segments


def _discretize(seg: np.ndarray, kind: str) -> np.ndarray:
    if kind == "bend":
        return seg
    length = np.linalg.norm(seg[1] - seg[0])
    n = max(1, int(np.ceil(length / ELEM_LEN_MM)))
    return seg[0] + np.linspace(0.0, 1.0, n + 1)[:, None] * (seg[1] - seg[0])


@dataclass
class FrameMesh:
    nodes: np.ndarray        # (n, 2)
    elements: np.ndarray     # (m, 2) node indices
    is_bend: np.ndarray      # (m,) bool
    is_gusset: np.ndarray    # (m,) bool
    tube_length: float       # total tube length of ONE rail + strut [mm]
    fixed_nodes: tuple[int, int]
    rider_node: int
    tail_node: int


def build_mesh(bend_radius_mm: float) -> FrameMesh:
    nodes: list[np.ndarray] = []
    kinds: list[bool] = []
    elems: list[tuple[int, int]] = []

    def add_node(p) -> int:
        for idx, q in enumerate(nodes):
            if np.hypot(*(q - p)) < 1e-6:
                return idx
        nodes.append(np.asarray(p, dtype=float))
        return len(nodes) - 1

    # rail: split first straight at the strut joint so the joint is a shared node
    segs = _fillet(_RAIL_CORNERS, bend_radius_mm)
    first, kind0 = segs[0]
    joint = np.array([_JOINT_X, 0.0])
    segs = [(np.vstack([first[0], joint]), kind0), (np.vstack([joint, first[1]]), kind0)] + segs[1:]
    prev = None
    for seg, kind in segs:
        pts = _discretize(seg, kind)
        for p in pts:
            idx = add_node(p)
            if prev is not None and idx != prev:
                elems.append((prev, idx))
                kinds.append(kind == "bend")
            prev = idx
    tail_node = prev
    joint_node = add_node(joint)
    # strut
    pts = _discretize(np.vstack([_STRUT_BASE, joint]), "straight")
    prev = None
    for p in pts:
        idx = add_node(p)
        if prev is not None:
            elems.append((prev, idx))
            kinds.append(False)
        prev = idx
    base_node = add_node(_STRUT_BASE)

    nodes_arr = np.array(nodes)
    elems_arr = np.array(elems)
    mids = nodes_arr[elems_arr].mean(axis=1)
    near_joint = np.hypot(*(mids - joint).T) < GUSSET_ZONE_MM
    near_mount = np.hypot(*(mids - _RAIL_CORNERS[0]).T) < GUSSET_ZONE_MM
    lengths = np.linalg.norm(nodes_arr[elems_arr[:, 1]] - nodes_arr[elems_arr[:, 0]], axis=1)
    rider_node = int(np.argmin(np.abs(nodes_arr[:, 0] - _RIDER_X) + 1e3 * (nodes_arr[:, 1] < -1e-6)))
    return FrameMesh(
        nodes=nodes_arr,
        elements=elems_arr,
        is_bend=np.array(kinds),
        is_gusset=near_joint | near_mount,
        tube_length=float(lengths.sum()),
        fixed_nodes=(add_node(_RAIL_CORNERS[0]), base_node),
        rider_node=rider_node,
        tail_node=int(tail_node),
    )


# ----------------------------------------------------------------------------- element matrices
def _frame_element(ea, ei, rho_a, length, c, s):
    l = length
    k = np.zeros((6, 6))
    k[[0, 3], [0, 3]] = ea / l
    k[0, 3] = k[3, 0] = -ea / l
    b = ei / l**3
    kb = b * np.array([[12, 6 * l, -12, 6 * l], [6 * l, 4 * l**2, -6 * l, 2 * l**2],
                       [-12, -6 * l, 12, -6 * l], [6 * l, 2 * l**2, -6 * l, 4 * l**2]])
    idx = [1, 2, 4, 5]
    k[np.ix_(idx, idx)] = kb
    m = np.zeros((6, 6))
    ma = rho_a * l / 6.0
    m[[0, 3], [0, 3]] = 2 * ma
    m[0, 3] = m[3, 0] = ma
    mb = rho_a * l / 420.0 * np.array([[156, 22 * l, 54, -13 * l], [22 * l, 4 * l**2, 13 * l, -3 * l**2],
                                        [54, 13 * l, 156, -22 * l], [-13 * l, -3 * l**2, -22 * l, 4 * l**2]])
    m[np.ix_(idx, idx)] = mb
    t = np.zeros((6, 6))
    r = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    t[:3, :3] = r
    t[3:, 3:] = r
    return k, m, t


# ----------------------------------------------------------------------------- high fidelity
_MESH_CACHE: dict[float, FrameMesh] = {}


def _mesh(radius: float) -> FrameMesh:
    key = round(float(radius), 6)
    if key not in _MESH_CACHE:
        if len(_MESH_CACHE) > 4096:
            _MESH_CACHE.clear()
        _MESH_CACHE[key] = build_mesh(key)
    return _MESH_CACHE[key]


def frame_fe_solve(tube_od_mm, wall_thickness_mm, gusset_thickness_mm, bend_radius_mm,
                   yield_strength_mpa, dynamic_load_factor) -> dict[str, float]:
    """High-fidelity frame FE solve for one design. Returns the 5 structural targets."""
    od, t, tg, rad = float(tube_od_mm), float(wall_thickness_mm), float(gusset_thickness_mm), float(bend_radius_mm)
    dlf = float(dynamic_load_factor)
    mesh = _mesh(rad)
    area, inertia, z_mod = section_properties(od, t)
    area, inertia, z_mod = float(area), float(inertia), float(z_mod)
    k_flex, i_sif = (float(v) for v in bend_factors(od, t, rad))
    kt_joint = float(joint_scf(t, tg))
    gusset_inertia = 0.3 * tg * 30.0**3 / 12.0  # partial in-plane contribution of the gusset plate

    n_dof = 3 * len(mesh.nodes)
    K = np.zeros((n_dof, n_dof))
    M = np.zeros((n_dof, n_dof))
    el_data = []
    for e, (a, b) in enumerate(mesh.elements):
        d = mesh.nodes[b] - mesh.nodes[a]
        length = float(np.hypot(*d))
        c, s = d / length
        ei = E_MPA * inertia
        if mesh.is_bend[e]:
            ei /= k_flex
        if mesh.is_gusset[e]:
            ei += E_MPA * gusset_inertia
        ke, me, te = _frame_element(E_MPA * area, ei, RHO_T_MM3 * area, length, c, s)
        dofs = np.r_[3 * a:3 * a + 3, 3 * b:3 * b + 3]
        K[np.ix_(dofs, dofs)] += te.T @ ke @ te
        M[np.ix_(dofs, dofs)] += te.T @ me @ te
        scf = 1.0
        if mesh.is_bend[e]:
            scf = max(scf, i_sif)
        if mesh.is_gusset[e]:
            scf = max(scf, kt_joint)
        el_data.append((dofs, ke, te, scf))

    # lumped payload + gusset masses
    for node, mass in ((mesh.rider_node, M_SEAT_T), (mesh.tail_node, M_TAIL_T)):
        M[3 * node, 3 * node] += mass
        M[3 * node + 1, 3 * node + 1] += mass

    fixed = np.r_[[3 * n + j for n in mesh.fixed_nodes for j in range(3)]]
    free = np.setdiff1d(np.arange(n_dof), fixed)

    # static: loads per rail
    F = np.zeros(n_dof)
    f_rider = dlf * P_RIDER_N / 2.0
    f_tail = dlf * P_TAIL_N / 2.0
    F[3 * mesh.rider_node + 1] -= f_rider
    F[3 * mesh.tail_node + 1] -= f_tail
    u = np.zeros(n_dof)
    u[free] = linalg.solve(K[np.ix_(free, free)], F[free], assume_a="pos")

    disp = np.hypot(u[0::3], u[1::3])
    torque = TORSION_SHARE * (f_rider + f_tail) * TORSION_ECC_MM
    tau = torque * (od / 2.0) / (2.0 * inertia)
    sigma_peak = 0.0
    for dofs, ke, te, scf in el_data:
        f_loc = ke @ (te @ u[dofs])
        axial = abs(f_loc[0]) / area
        moment = max(abs(f_loc[2]), abs(f_loc[5]))
        sigma = (axial + moment / z_mod) * scf
        sigma_peak = max(sigma_peak, float(np.sqrt(sigma**2 + 3.0 * tau**2)))

    # modal
    lam = linalg.eigh(K[np.ix_(free, free)], M[np.ix_(free, free)], subset_by_index=[0, 0],
                      eigvals_only=True)[0]
    f1 = float(np.sqrt(max(lam, 0.0)) / (2.0 * np.pi))

    gusset_mass = 2 * GUSSET_AREA_MM2 * tg * RHO_KG_MM3  # 2 gussets per rail
    mass = 2.0 * (mesh.tube_length * area * RHO_KG_MM3 + gusset_mass) \
        + CROSS_MEMBER_LEN_MM * area * RHO_KG_MM3 + BRACKETS_KG

    return {
        "mass_kg": float(mass),
        "peak_von_mises_mpa": sigma_peak,
        "max_deflection_mm": float(disp.max()),
        "first_mode_hz": f1,
        "log10_fatigue_life_cycles": float(fatigue_log10_life(sigma_peak, yield_strength_mpa, dlf)),
    }


def frame_fe_batch(X: np.ndarray) -> np.ndarray:
    """Run ``frame_fe_solve`` row-wise. X columns follow ``config.FEATURES``. Returns (n, 5)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    out = np.empty((len(X), 5))
    for r, row in enumerate(X):
        res = frame_fe_solve(*row)
        out[r] = [res["mass_kg"], res["peak_von_mises_mpa"], res["max_deflection_mm"],
                  res["first_mode_hz"], res["log10_fatigue_life_cycles"]]
    return out


# ----------------------------------------------------------------------------- low fidelity
# Calibration constants: chosen so the hand calc matches the FE model at the design-space centre.
_CF_LEN_MM = 2.0 * _STRAIGHT_LEN_MM + CROSS_MEMBER_LEN_MM   # straight-line tube length, no bends
_CF_STRESS_C = 1.0                     # deliberately uncalibrated: nominal (no-SCF) stress
_CF_DEFL_C = 0.789
_CF_FREQ_C = 0.975
_CF_SPAN_MM = _TAIL_X - _JOINT_X       # overhang beyond the strut joint


def closed_form(X: np.ndarray) -> np.ndarray:
    """Vectorised low-fidelity hand calculation. X columns follow ``config.FEATURES``."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    od, t, tg, rad, sy, dlf = X.T
    area, inertia, z_mod = section_properties(od, t)
    f_total = dlf * (P_RIDER_N + P_TAIL_N) / 2.0
    moment = _CF_STRESS_C * dlf * (P_RIDER_N * (_RIDER_X - _JOINT_X) + P_TAIL_N * _CF_SPAN_MM) / 2.0
    sigma = moment / z_mod                     # nominal stress: no SCF / SIF (classic hand calc)
    tau = TORSION_SHARE * f_total * TORSION_ECC_MM * (od / 2.0) / (2.0 * inertia)
    vm = np.sqrt(sigma**2 + 3.0 * tau**2)
    defl = _CF_DEFL_C * f_total * _CF_SPAN_MM**3 / (3.0 * E_MPA * inertia)
    k_stiff = 3.0 * E_MPA * inertia / _CF_SPAN_MM**3
    m_eff = M_SEAT_T + M_TAIL_T + 0.24 * _STRAIGHT_LEN_MM * area * RHO_T_MM3
    f1 = _CF_FREQ_C * np.sqrt(k_stiff / m_eff) / (2.0 * np.pi)
    mass = _CF_LEN_MM * area * RHO_KG_MM3 + 4 * GUSSET_AREA_MM2 * tg * RHO_KG_MM3 + BRACKETS_KG
    return np.column_stack([mass, vm, defl, f1, fatigue_log10_life(vm, sy, dlf)])


def safety_factor(yield_mpa, peak_vm):
    return np.asarray(yield_mpa, dtype=float) / np.clip(np.asarray(peak_vm, dtype=float), 1e-6, None)


__all__ = ["FEATURES", "frame_fe_solve", "frame_fe_batch", "closed_form", "section_properties",
           "bend_factors", "joint_scf", "fatigue_log10_life", "safety_factor", "build_mesh"]

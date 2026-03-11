"""
energy_sweep.py

Dynamic cyclic loading simulation for the laminated quarter-elliptic
cantilever spike model. Sweeps hinge stiffness and computes energy
storage, dissipation, propulsion work, and mechanical efficiency.

Model:  models/leafspring_final.xml
Output: outputs/energy_sweep.csv
"""

import os
import csv
import re
import numpy as np
import mujoco

# ---------------------------------------------------------------------------
# Paths and sweep parameters
# ---------------------------------------------------------------------------

XML_PATH = "models/leafspring_final.xml"
OUT_CSV  = "outputs/energy_sweep.csv"

K_HINGE_VALUES = np.linspace(0.001, 0.05, 20)

# ---------------------------------------------------------------------------
# Force and timing parameters
# ---------------------------------------------------------------------------

F_MAX     = 0.1   # N  (applied downward, stored as negative z in xfrc_applied)
T_SETTLE  = 0.5   # s  zero-force settle before the loading cycle
T_RAMP_UP = 1.0   # s
T_HOLD    = 0.5   # s
T_RAMP_DN = 1.0   # s

F_STATIC  = 0.1   # N  for static stiffness calibration
T_STATIC  = 2.0   # s  settle time for static test


# ---------------------------------------------------------------------------

def load_model_with_stiffness(xml_path: str, k_hinge: float) -> mujoco.MjModel:
    """
    Read the XML file and replace every hinge stiffness attribute with
    k_hinge, then compile and return the MjModel.
    """
    with open(xml_path, "r") as f:
        xml = f.read()

    xml = re.sub(r'stiffness="[^"]*"', f'stiffness="{k_hinge:.8f}"', xml)
    return mujoco.MjModel.from_xml_string(xml)


# ---------------------------------------------------------------------------

def run_static_test(model: mujoco.MjModel):
    """
    Apply F_STATIC downward at the leaf1 tip for T_STATIC seconds and
    measure the equilibrium vertical deflection.

    Returns
    -------
    delta : float  vertical deflection in metres (downward positive)
    k_eq  : float  equivalent stiffness F/delta  [N/m]
    """
    data = mujoco.MjData(model)
    dt   = model.opt.timestep

    tip_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "leaf1_tip")
    tip_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_site")

    if tip_body < 0 or tip_site < 0:
        raise RuntimeError("'leaf1_tip' body or 'tip_site' site not found in model.")

    mujoco.mj_forward(model, data)
    z0 = float(data.site_xpos[tip_site][2])

    steps = int(T_STATIC / dt)
    for _ in range(steps):
        data.xfrc_applied[:] = 0.0
        data.xfrc_applied[tip_body, 2] = +F_STATIC
        mujoco.mj_step(model, data)

    mujoco.mj_forward(model, data)
    z1 = float(data.site_xpos[tip_site][2])

    delta = z1 - z0   # positive = upward displacement

    if abs(delta) < 1e-12:
        return 0.0, np.nan

    k_eq = F_STATIC / delta
    return delta, k_eq


# ---------------------------------------------------------------------------

def run_cycle_simulation(model: mujoco.MjModel) -> list:
    """
    Run one settle + load/unload cycle on a fresh MjData.

    Force profile (upward, +z):
        [0, T_SETTLE)          : F = 0          (settle)
        [0, T_RAMP_UP)         : ramp 0 → F_MAX
        [T_RAMP_UP, +T_HOLD)   : hold F_MAX
        [+T_HOLD, +T_RAMP_DN)  : ramp F_MAX → 0

    Each timestep records:
        t, Fz, delta, vx, Fx, joint_vels

    Fx is the x-component of the Cartesian elastic force at the tip site,
    computed via the Jacobian pseudo-inverse:
        f_cart = (J J^T)^{-1} J q_spring   (virtual-work projection)

    Returns
    -------
    records : list of dicts, one per timestep
    """
    data = mujoco.MjData(model)
    dt   = model.opt.timestep

    tip_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "leaf1_tip")
    tip_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_site")

    # ---- settle phase (zero force) ----
    mujoco.mj_forward(model, data)
    settle_steps = int(T_SETTLE / dt)
    for _ in range(settle_steps):
        data.xfrc_applied[:] = 0.0
        mujoco.mj_step(model, data)

    mujoco.mj_forward(model, data)
    z0 = float(data.site_xpos[tip_site][2])   # baseline after settling

    # ---- loading cycle ----
    cycle_time  = T_RAMP_UP + T_HOLD + T_RAMP_DN
    cycle_steps = int(cycle_time / dt)
    records     = []

    # pre-allocate Jacobian buffer (reused each step)
    jac_pos = np.zeros((3, model.nv))
    reg_eye = 1e-12 * np.eye(3)   # Tikhonov regularisation

    for i in range(cycle_steps):
        t = i * dt

        # -- force profile --
        if t < T_RAMP_UP:
            Fz = +F_MAX * (t / T_RAMP_UP)
        elif t < T_RAMP_UP + T_HOLD:
            Fz = +F_MAX
        else:
            s  = (t - T_RAMP_UP - T_HOLD) / T_RAMP_DN
            Fz = +F_MAX * (1.0 - min(s, 1.0))

        data.xfrc_applied[:] = 0.0
        data.xfrc_applied[tip_body, 2] = Fz

        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)

        # -- tip deflection --
        z_tip = float(data.site_xpos[tip_site][2])
        delta  = z_tip - z0    # upward positive

        # -- tip x-velocity (body spatial velocity: [ang(3), lin(3)]) --
        vx = float(data.cvel[tip_body][3])

        # -- horizontal elastic force at tip via Jacobian pseudo-inverse --
        # Virtual work: tau = J^T f  →  f = (J J^T)^{-1} J tau
        mujoco.mj_jacSite(model, data, jac_pos, None, tip_site)
        J   = jac_pos                          # 3 × nv
        tau = data.qfrc_spring                 # nv  (spring generalized forces)
        JJT = J @ J.T                          # 3 × 3
        try:
            f_cart = np.linalg.solve(JJT + reg_eye, J @ tau)
            Fx = float(f_cart[0])
        except np.linalg.LinAlgError:
            Fx = 0.0

        records.append({
            "t":          t,
            "Fz":         Fz,
            "delta":      delta,
            "vx":         vx,
            "Fx":         Fx,
            "joint_vels": data.qvel.copy(),
        })

    return records


# ---------------------------------------------------------------------------

def compute_energy_metrics(records: list, model: mujoco.MjModel, k_eq: float):
    """
    Compute scalar energy quantities from the per-step simulation record.

    Parameters
    ----------
    records : list of dicts from run_cycle_simulation
    model   : MjModel (needed for dof_damping and timestep)
    k_eq    : equivalent stiffness [N/m] from static calibration

    Returns
    -------
    delta_max  : float  maximum tip deflection [m]
    U_max      : float  stored elastic energy at max deflection [J]
    E_diss     : float  total dissipated energy over the cycle [J]
    W_prop     : float  propulsion work ∫ Fx · vx dt [J]
    efficiency : float  η = W_prop / (W_prop + E_diss)
    """
    dt          = model.opt.timestep
    dof_damping = model.dof_damping   # array, shape (nv,)

    deltas = np.array([r["delta"] for r in records])
    vxs    = np.array([r["vx"]    for r in records])
    Fxs    = np.array([r["Fx"]    for r in records])

    # Maximum deflection
    delta_max = float(np.max(deltas))

    # Peak stored elastic energy
    U_max = 0.5 * k_eq * delta_max ** 2

    # Dissipated energy: E_diss = ∫ Σ_j  c_j · q̇_j²  dt
    E_diss = 0.0
    for r in records:
        qvel    = r["joint_vels"]
        E_diss += float(np.dot(dof_damping, qvel ** 2)) * dt

    # Propulsion work: W_prop = ∫ Fx · vx  dt
    W_prop = float(np.sum(Fxs * vxs) * dt)

    # Mechanical efficiency
    denom      = W_prop + E_diss
    efficiency = (W_prop / denom) if abs(denom) > 1e-15 else 0.0

    return delta_max, U_max, E_diss, W_prop, efficiency


# ---------------------------------------------------------------------------

def main():
    os.makedirs("outputs", exist_ok=True)

    header = ["k_hinge", "k_equivalent", "delta_max",
              "U_max", "E_diss", "W_prop", "efficiency"]
    rows   = []

    print(f"{'k_hinge':>10} {'k_eq':>10} {'δ_max mm':>10} "
          f"{'U_max μJ':>10} {'E_diss μJ':>10} "
          f"{'W_prop μJ':>10} {'η':>8}")
    print("-" * 75)

    for k_hinge in K_HINGE_VALUES:

        # --- load model ---
        model = load_model_with_stiffness(XML_PATH, k_hinge)

        # --- static calibration → k_eq ---
        _, k_eq = run_static_test(model)

        if np.isnan(k_eq) or k_eq <= 0:
            print(f"{k_hinge:10.5f}  SKIP (zero static deflection)")
            continue

        # --- reload fresh model for cycle (clean MjData state) ---
        model = load_model_with_stiffness(XML_PATH, k_hinge)

        # --- dynamic cycle ---
        records = run_cycle_simulation(model)

        # --- energy metrics ---
        delta_max, U_max, E_diss, W_prop, efficiency = compute_energy_metrics(
            records, model, k_eq
        )

        rows.append([k_hinge, k_eq, delta_max, U_max, E_diss, W_prop, efficiency])

        print(f"{k_hinge:10.5f} {k_eq:10.4f} {delta_max*1e3:10.3f} "
              f"{U_max*1e6:10.3f} {E_diss*1e6:10.3f} "
              f"{W_prop*1e6:10.3f} {efficiency:8.4f}")

    # --- write CSV ---
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows → {OUT_CSV}")
    print("\nTo plot results:")
    print("  import pandas as pd, matplotlib.pyplot as plt")
    print(f"  df = pd.read_csv('{OUT_CSV}')")
    print("  df.plot(x='k_hinge', y=['delta_max','U_max','efficiency'])")
    print("  plt.show()")


if __name__ == "__main__":
    main()

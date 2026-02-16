import csv
import os
import re
import numpy as np
import mujoco

XML_PATH = "models/leafspring_final.xml"
OUT_CSV  = "outputs/k_sweep_results.csv"

F_MAX = 0.1          # N
DELTA_LIMIT = 0.005  # 5 mm constraint
DT = 0.001
T_RAMP = 1.0
T_HOLD = 0.3
T_REL  = 1.0

K_VALUES = np.linspace(5, 80, 20)


def load_model_with_k(xml_path, k):
    with open(xml_path, "r") as f:
        xml = f.read()

    xml = re.sub(
        r'stiffness="[^"]*"',
        f'stiffness="{k}"',
        xml
    )

    return mujoco.MjModel.from_xml_string(xml)


def compute_stored_energy(model, data, k):
    U = 0.0
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            qadr = model.jnt_qposadr[j]
            theta = data.qpos[qadr]
            U += 0.5 * k * theta**2
    return U


def run_simulation(k):

    model = load_model_with_k(XML_PATH, k)
    data  = mujoco.MjData(model)

    tip_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "leaf1_tip")
    tip_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_site")

    mujoco.mj_forward(model, data)
    z0 = float(data.site_xpos[tip_site][2])

    ramp_steps = int(T_RAMP / DT)
    hold_steps = int(T_HOLD / DT)
    rel_steps  = int(T_REL  / DT)

    dissipated = 0.0
    delta_max = 0.0
    U_max = 0.0

    # ---- Ramp ----
    for i in range(ramp_steps):
        Fz = -F_MAX * (i / ramp_steps)

        data.xfrc_applied[:] = 0
        data.xfrc_applied[tip_body, 2] = Fz

        mujoco.mj_step(model, data)

        z = float(data.site_xpos[tip_site][2])
        delta = z0 - z
        delta_max = max(delta_max, delta)

        U = compute_stored_energy(model, data, k)
        U_max = max(U_max, U)

        for j in range(model.nv):
            dissipated += model.dof_damping[j] * data.qvel[j]**2 * DT

    # ---- Hold ----
    for _ in range(hold_steps):
        data.xfrc_applied[:] = 0
        data.xfrc_applied[tip_body, 2] = -F_MAX
        mujoco.mj_step(model, data)

    # ---- Release ----
    for _ in range(rel_steps):
        data.xfrc_applied[:] = 0
        mujoco.mj_step(model, data)

        for j in range(model.nv):
            dissipated += model.dof_damping[j] * data.qvel[j]**2 * DT

    # ---- Constraint ----
    feasible = delta_max <= DELTA_LIMIT

    if feasible:
        eta = max(0.0, 1.0 - dissipated / (U_max + 1e-12))
        J = U_max - dissipated
    else:
        eta = 0.0
        J = -1.0  # penalize infeasible

    return delta_max, U_max, dissipated, eta, J, feasible


def main():
    os.makedirs("outputs", exist_ok=True)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "k",
            "delta_max_m",
            "U_max_J",
            "E_diss_J",
            "efficiency",
            "objective_J",
            "feasible"
        ])

        for k in K_VALUES:
            delta, U, Ed, eta, J, feasible = run_simulation(k)

            writer.writerow([k, delta, U, Ed, eta, J, feasible])

            print(
                f"k={k:.2f} | "
                f"delta={delta*1000:.2f} mm | "
                f"U={U:.2e} | "
                f"eta={eta:.3f} | "
                f"{'OK' if feasible else 'OVER-DEFLECT'}"
            )

    print(f"\nSaved results to {OUT_CSV}")


if __name__ == "__main__":
    main()

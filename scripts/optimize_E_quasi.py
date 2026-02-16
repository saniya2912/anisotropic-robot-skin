import csv
import os
import re
import numpy as np
import mujoco

XML_PATH = "models/leafspring.xml"
OUT_CSV  = "outputs/k_sweep_quasistatic.csv"

F_MAX = 0.1
DELTA_LIMIT = 0.005

DT = 0.001
N_STEPS = 100
SETTLE_STEPS = 200

K_VALUES = np.linspace(5, 80, 20)


def load_model_with_k(xml_path, k):
    with open(xml_path, "r") as f:
        xml = f.read()

    xml = re.sub(r'stiffness="[^"]*"', f'stiffness="{k}"', xml)

    return mujoco.MjModel.from_xml_string(xml)


def settle_system(model, data, steps):
    for _ in range(steps):
        mujoco.mj_step(model, data)


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

    # ---- Loading ----
    for i in range(N_STEPS):
        F = -F_MAX * (i / N_STEPS)

        data.xfrc_applied[:] = 0
        data.xfrc_applied[tip_body, 2] = F

        settle_system(model, data, SETTLE_STEPS)

    z_loaded = float(data.site_xpos[tip_site][2])
    delta_max = z0 - z_loaded
    U_input = compute_stored_energy(model, data, k)

    # ---- Unloading ----
    for i in range(N_STEPS):
        F = -F_MAX * (1 - i / N_STEPS)

        data.xfrc_applied[:] = 0
        data.xfrc_applied[tip_body, 2] = F

        settle_system(model, data, SETTLE_STEPS)

    z_final = float(data.site_xpos[tip_site][2])
    U_recovered = compute_stored_energy(model, data, k)

    # ---- Efficiency ----
    if U_input > 1e-12:
        eta = max(0.0, U_recovered / U_input)
    else:
        eta = 0.0

    J = U_input if delta_max <= DELTA_LIMIT else -1.0

    return delta_max, U_input, eta, J


def main():
    os.makedirs("outputs", exist_ok=True)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "delta_max_m", "U_input_J", "efficiency", "objective_J"])

        for k in K_VALUES:
            delta, U, eta, J = run_simulation(k)

            writer.writerow([k, delta, U, eta, J])

            print(
                f"k={k:.2f} | "
                f"delta={delta*1000:.3f} mm | "
                f"U={U:.2e} | "
                f"eta={eta:.3f}"
            )

    print("\nSaved results.")


if __name__ == "__main__":
    main()

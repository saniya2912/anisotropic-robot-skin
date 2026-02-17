import csv
import os
import re
import numpy as np
import mujoco

XML_PATH = "models/leafspring_final.xml"
OUT_CSV  = "outputs/stiffness_calibration.csv"

F_STATIC = 0.1     # Newton
DT       = 0.001
T_SETTLE = 2.0     # seconds

# Sweep hinge stiffness values
KH_VALUES = np.linspace(0.001, 0.05, 12)


def load_model_with_kh(xml_path, kh):
    with open(xml_path, "r") as f:
        xml = f.read()

    # Replace all hinge stiffness values
    xml = re.sub(
        r'stiffness="[^"]*"',
        f'stiffness="{kh}"',
        xml
    )

    return mujoco.MjModel.from_xml_string(xml)


def run_static_test(kh):

    model = load_model_with_kh(XML_PATH, kh)
    data  = mujoco.MjData(model)

    tip_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "leaf1_tip")
    tip_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_site")

    if tip_body < 0 or tip_site < 0:
        raise RuntimeError("Tip body or tip_site not found in XML.")

    mujoco.mj_forward(model, data)
    z0 = float(data.site_xpos[tip_site][2])

    steps = int(T_SETTLE / DT)

    for _ in range(steps):
        data.xfrc_applied[:] = 0
        data.xfrc_applied[tip_body, 2] = -F_STATIC
        mujoco.mj_step(model, data)

    mujoco.mj_forward(model, data)
    z1 = float(data.site_xpos[tip_site][2])

    delta = z0 - z1

    if abs(delta) < 1e-12:
        k_eq = np.nan
    else:
        k_eq = F_STATIC / delta

    return delta, k_eq


def main():
    os.makedirs("outputs", exist_ok=True)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k_hinge", "delta_m", "k_equivalent"])

        for kh in KH_VALUES:
            delta, k_eq = run_static_test(kh)
            writer.writerow([kh, delta, k_eq])

            print(f"k_h={kh:.5f}  "
                  f"delta={delta*1000:.3f} mm  "
                  f"k_eq={k_eq:.3f} N/m")

    print(f"\nSaved results to {OUT_CSV}")


if __name__ == "__main__":
    main()

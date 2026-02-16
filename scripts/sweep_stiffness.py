import csv
import numpy as np
import mujoco

XML_PATH = "models/leafspring.xml"
OUT_CSV  = "outputs/stiffness_sweep.csv"

# Force applied at tip
Fz = -0.2   # N (downward)

# Stiffness values to test
k_values = np.linspace(0.0005, 0.01, 10)

def load_model_with_k(k):
    with open(XML_PATH, "r") as f:
        xml = f.read()

    # replace stiffness in leaf1 hinge
    import re
    xml = re.sub(r'stiffness="[^"]*"', f'stiffness="{k}"', xml)

    return mujoco.MjModel.from_xml_string(xml)

def measure_deflection(model):
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    tip_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tip")
    tip_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_site")

    mujoco.mj_forward(model, data)
    z0 = data.site_xpos[tip_site][2]

    for _ in range(int(2.0 / dt)):
        data.xfrc_applied[:] = 0
        data.xfrc_applied[tip_body, 2] = Fz
        mujoco.mj_step(model, data)

    mujoco.mj_forward(model, data)
    z1 = data.site_xpos[tip_site][2]

    return z0 - z1  # deflection

def main():
    results = []

    for k in k_values:
        model = load_model_with_k(k)
        delta = measure_deflection(model)
        k_eff = abs(Fz) / delta if delta > 1e-9 else np.nan

        print(f"k_input={k:.6f}  delta={delta*1000:.3f} mm  k_eff={k_eff:.6f}")
        results.append([k, delta, k_eff])

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k_input", "delta_m", "k_effective"])
        writer.writerows(results)

    print("Saved:", OUT_CSV)

if __name__ == "__main__":
    main()

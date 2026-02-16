import csv
import numpy as np
import mujoco

XML_PATH = "models/leafspring.xml"
OUT_CSV  = "outputs/dynamic_energy.csv"

F_max = -0.2  # peak downward force
T_down = 1.0
T_up   = 1.0

def main(k=0.005, damping=0.001):

    # modify XML stiffness and damping
    with open(XML_PATH, "r") as f:
        xml = f.read()

    import re
    xml = re.sub(r'stiffness="[^"]*"', f'stiffness="{k}"', xml)
    xml = re.sub(r'damping="[^"]*"', f'damping="{damping}"', xml)

    model = mujoco.MjModel.from_xml_string(xml)
    data  = mujoco.MjData(model)
    dt = model.opt.timestep

    tip_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tip")
    tip_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_site")

    mujoco.mj_forward(model, data)
    z0 = data.site_xpos[tip_site][2]

    dissipated = 0.0
    t = 0.0

    rows = []

    total_time = T_down + T_up
    steps = int(total_time / dt)

    for i in range(steps):

        # smooth force ramp
        if t < T_down:
            s = t / T_down
            Fz = F_max * s
        else:
            s = (t - T_down) / T_up
            Fz = F_max * (1 - s)

        data.xfrc_applied[:] = 0
        data.xfrc_applied[tip_body, 2] = Fz

        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)

        z = data.site_xpos[tip_site][2]
        delta = z0 - z

        # approximate vertical velocity of tip
        vz = data.cvel[tip_body][5]  # z linear velocity

        U_store = 0.5 * k * delta**2
        dissipated += damping * vz**2 * dt

        rows.append([t, Fz, delta, vz, U_store, dissipated])

        t += dt

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "Fz", "delta", "v_tip", "U_store", "E_diss"])
        writer.writerows(rows)

    print("Saved:", OUT_CSV)

if __name__ == "__main__":
    main()

import time
import numpy as np
import mujoco
import mujoco.viewer

XML_PATH = "models/leafspring.xml"

def main():

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    dt = model.opt.timestep

    # -----------------------------
    # Locate tip body
    # -----------------------------
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "leaf4")
    if body_id < 0:
        raise RuntimeError("Body 'leaf4' not found.")

    # Locate tip site
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_site")
    if site_id < 0:
        raise RuntimeError("Site 'tip_site' not found.")

    # Initial tip height
    mujoco.mj_forward(model, data)
    z_initial = data.site_xpos[site_id][2]

    print("Initial tip height:", z_initial)

    # -----------------------------
    # Apply constant downward force
    # -----------------------------
    Fz = 1  # 0.05 N downward (small load for mm-scale system)

    settle_time = 3.0
    steps = int(settle_time / dt)

    with mujoco.viewer.launch_passive(model, data) as viewer:

        viewer.cam.distance = 0.05
        viewer.cam.lookat[:] = [0.01, 0.0, 0.03]
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 90

        for i in range(steps):

            # Clear previous forces
            data.xfrc_applied[:] = 0

            # Apply vertical force to tip body
            data.xfrc_applied[body_id, 2] = Fz

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(dt)

        # Measure final deflection
        z_final = data.site_xpos[site_id][2]
        delta = z_initial - z_final

        print("Final tip height:", z_final)
        print("Deflection (m):", delta)
        print("Deflection (mm):", delta * 1000)

        # Hold final state
        while viewer.is_running():
            viewer.sync()
            time.sleep(dt)

if __name__ == "__main__":
    main()

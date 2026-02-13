import time
import numpy as np
import mujoco
import mujoco.viewer

XML_PATH = "models/leafspring.xml"

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    dt = model.opt.timestep

    # Apply force to the longest leaf body
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "leaf1")
    if body_id < 0:
        raise RuntimeError("Body 'leaf1' not found (longest leaf).")

    # Measure deflection at the tip site (must be added to leaf1 in XML)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_site")
    if site_id < 0:
        raise RuntimeError(
            "Site 'tip_site' not found. Add <site name='tip_site' pos='0.016 0 0' ...> inside leaf1."
        )

    mujoco.mj_forward(model, data)
    z_initial = float(data.site_xpos[site_id][2])
    x_initial = float(data.site_xpos[site_id][0])

    print("Initial tip position:", data.site_xpos[site_id])

    # Downward point load
    Fz = 0.5  # N (increase if you want more visible deflection)

    settle_time = 3.0
    steps = int(settle_time / dt)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 0.08
        viewer.cam.lookat[:] = [0.008, 0.0, 0.03]
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 90

        for _ in range(steps):
            data.xfrc_applied[:] = 0.0

            # Apply force at the COM of leaf1 (MuJoCo API limitation)
            # This is fine for now because we measure tip deflection.
            data.xfrc_applied[body_id, 2] = Fz

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(dt)

        mujoco.mj_forward(model, data)
        z_final = float(data.site_xpos[site_id][2])
        delta = z_initial - z_final

        print("Final tip position:", data.site_xpos[site_id])
        print("Deflection (mm):", delta * 1000.0)

        # Hold
        while viewer.is_running():
            viewer.sync()
            time.sleep(dt)

if __name__ == "__main__":
    main()

import time
import mujoco
import mujoco.viewer

XML_PATH = "models/leafspring_final.xml"

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    dt = model.opt.timestep

    # Apply force to the TIP body (true point load at the correct location)
    tip_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "leaf1_tip")
    if tip_body < 0:
        raise RuntimeError("Body 'tip' not found. Did you add the tip body inside leaf1 in XML?")

    tip_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_site")
    if tip_site < 0:
        raise RuntimeError("Site 'tip_site' not found. Did you add it to the tip body?")

    mujoco.mj_forward(model, data)
    tip0 = data.site_xpos[tip_site].copy()
    print("Initial tip position:", tip0)

    # Downward load (NEGATIVE z)
    Fz = 0.1  # N (increase magnitude for more visible bend, e.g. -1.0)

    # Let it settle under load
    settle_time = 3.0
    steps = int(settle_time / dt)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 0.09
        viewer.cam.lookat[:] = [0.010, 0.0, 0.03]
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 90

        for _ in range(steps):
            # Clear forces every step
            data.xfrc_applied[:] = 0.0

            # Apply force at tip body COM (COM is at the tip location because body pos is at tip)
            data.xfrc_applied[tip_body, 2] = Fz

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(dt)

        mujoco.mj_forward(model, data)
        tip1 = data.site_xpos[tip_site].copy()

        dz = float(tip0[2] - tip1[2])
        print("Final tip position:", tip1)
        print("Deflection dz (mm):", dz * 1000.0)

        # Hold view
        while viewer.is_running():
            viewer.sync()
            time.sleep(dt)

if __name__ == "__main__":
    main()

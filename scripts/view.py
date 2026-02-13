import time
import mujoco
import mujoco.viewer

XML_PATH = "models/leafspring.xml"

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    dt = model.opt.timestep

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 0.05
        viewer.cam.lookat[:] = [0.0, 0.0, 0.01]
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 90

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(dt)

if __name__ == "__main__":
    main()

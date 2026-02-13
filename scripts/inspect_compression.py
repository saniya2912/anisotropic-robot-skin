import time
import numpy as np
import mujoco
import mujoco.viewer

XML_PATH = "models/skin_deformable.xml"

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "top_servo")
    if act_id < 0:
        raise RuntimeError("Actuator 'top_servo' not found. Check XML.")

    # gentle squeeze range in joint coords (0..0.04)
    q_start = 0.000   # plate up
    q_end   = 0.020   # squeeze down 2cm (adjust later)

    # go slow (quasi-static)
    t_down, t_hold, t_up = 3.0, 0.5, 3.0
    n_down = int(t_down / dt)
    n_hold = int(t_hold / dt)
    n_up   = int(t_up / dt)

    def ease(a, b, n):
        s = np.linspace(0, 1, n)
        s = 0.5 * (1 - np.cos(np.pi * s))
        return (1 - s) * a + s * b

    traj = []
    traj += list(ease(q_start, q_end, n_down))
    traj += [q_end] * n_hold
    traj += list(ease(q_end, q_start, n_up))
    traj += [q_start] * n_hold

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 0.7
        viewer.cam.lookat[:] = [0.0, 0.0, 0.05]
        viewer.cam.elevation = -25
        viewer.cam.azimuth = 90

        # settle
        for _ in range(int(0.5 / dt)):
            data.ctrl[act_id] = q_start
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(dt)

        while viewer.is_running():
            for q in traj:
                if not viewer.is_running():
                    break
                data.ctrl[act_id] = float(q)
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(dt)

if __name__ == "__main__":
    main()

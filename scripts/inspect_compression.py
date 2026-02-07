import time
import numpy as np
import mujoco
import mujoco.viewer

XML_PATH = "models/skin_deformable.xml"

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # --- Find actuator and joint by name (robust) ---
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "plate_servo")
    if act_id < 0:
        raise RuntimeError(
            "Actuator 'plate_servo' not found. "
            "Make sure your XML actuator has name='plate_servo'."
        )

    j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "plate_slide")
    if j_id < 0:
        raise RuntimeError("Joint 'plate_slide' not found. Check your XML joint name.")

    qpos_adr = model.jnt_qposadr[j_id]
    z0 = float(data.qpos[qpos_adr])

    # How far to compress (meters)
    z_down = 0.08
    zmin = max(0.0, z0 - z_down)

    dt = model.opt.timestep

    # Smooth cyclic motion parameters
    period = 2.5  # seconds per full down-up cycle
    omega = 2.0 * np.pi / period

    # --- Launch viewer ---
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 0.5
        viewer.cam.lookat[:] = [0.0, 0.0, 0.12]
        viewer.cam.elevation = -25
        viewer.cam.azimuth = 90

        t0_wall = time.time()
        step = 0

        while viewer.is_running():
            t = time.time() - t0_wall

            # Sinusoid in [0, 1] then map to [z0, zmin]
            s = 0.5 * (1.0 - np.cos(omega * t))  # 0 -> 1 -> 0
            target = (1.0 - s) * z0 + s * zmin

            # Position servo target
            data.ctrl[act_id] = float(target)

            mujoco.mj_step(model, data)
            viewer.sync()

            # Optional: print occasionally
            if step % int(0.5 / dt) == 0:
                print(f"t={t:5.2f}s  plate_target={target: .4f}  plate_qpos={data.qpos[qpos_adr]: .4f}")

            step += 1
            time.sleep(dt)

if __name__ == "__main__":
    main()

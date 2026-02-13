import csv
import os
import time
import re
import numpy as np
import mujoco
import mujoco.viewer

XML_PATH = "models/spike_spring_test.xml"
OUT_CSV  = "outputs/spike_test_log.csv"

# --- You can tweak these safely ---
FN_TOUCH_N = 0.3         # N threshold to say "we made contact"
EXTRA_COMPRESS = 0.002   # 2mm extra compression after touch
MAX_APPROACH = 0.006     # max indenter travel during search for contact
T_APPROACH = 2.0         # seconds
T_COMPRESS = 2.0         # seconds
T_HOLD = 0.3             # seconds
T_RELEASE = 2.0          # seconds
PRINT_EVERY_S = 0.2      # debug print interval
# -------------------------------


def load_model_with_spike_params(xml_path: str, k: float, c: float) -> mujoco.MjModel:
    """Edit XML text to set stiffness/damping on joint spike_z, then compile."""
    with open(xml_path, "r", encoding="utf-8") as f:
        xml = f.read()

    def repl(match):
        tag = match.group(0)
        # remove existing stiffness/damping if present
        tag = re.sub(r'\sstiffness="[^"]*"', "", tag)
        tag = re.sub(r'\sdamping="[^"]*"', "", tag)

        # insert stiffness + damping before close
        if tag.endswith("/>"):
            tag = tag[:-2] + f' stiffness="{k}" damping="{c}"/>'
        else:
            tag = tag[:-1] + f' stiffness="{k}" damping="{c}">'
        return tag

    xml_new, n = re.subn(r'<joint\b[^>]*\bname="spike_z"[^>]*\/?>', repl, xml, count=1)
    if n != 1:
        raise RuntimeError("Could not find joint name='spike_z' in XML.")

    return mujoco.MjModel.from_xml_string(xml_new)


def contact_normal_force_between(model, data, geomA, geomB) -> float:
    """Sum of normal forces for contacts between two named geoms."""
    gA = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geomA)
    gB = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geomB)
    if gA < 0 or gB < 0:
        raise RuntimeError(f"Geom not found: {geomA} or {geomB}")

    Fn = 0.0
    cf = np.zeros(6, dtype=float)
    for i in range(data.ncon):
        con = data.contact[i]
        if (con.geom1 == gA and con.geom2 == gB) or (con.geom1 == gB and con.geom2 == gA):
            mujoco.mj_contactForce(model, data, i, cf)
            Fn += cf[0]
    return Fn


def ease_traj(a: float, b: float, n: int):
    """Cosine ease-in-out from a to b."""
    n = max(2, int(n))
    s = np.linspace(0, 1, n)
    s = 0.5 * (1 - np.cos(np.pi * s))
    return list((1 - s) * a + s * b)


def main(k=5000.0, c=2.0, view=True):
    model = load_model_with_spike_params(XML_PATH, k, c)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "indenter_servo")
    if act < 0:
        raise RuntimeError("Actuator 'indenter_servo' not found")

    j_spike = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "spike_z")
    if j_spike < 0:
        raise RuntimeError("Joint 'spike_z' not found")
    qadr_spike = model.jnt_qposadr[j_spike]
    dadr_spike = model.jnt_dofadr[j_spike]

    # Start indenter at 0 joint coordinate (up)
    q_start = 0.0

    # Prepare CSV
    os.makedirs("outputs", exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "phase", "indenter_target", "x_spike", "xdot_spike",
                    "Fn_tip_indenter", "E_store", "E_diss_cum"])

        dissipated = 0.0

        def step_once(target, t, phase):
            nonlocal dissipated
            data.ctrl[act] = float(target)
            mujoco.mj_step(model, data)

            x = float(data.qpos[qadr_spike])      # spike compression coordinate (m)
            xdot = float(data.qvel[dadr_spike])   # m/s
            Fn = contact_normal_force_between(model, data, "tip_geom", "indenter_geom")

            E_store = 0.5 * k * x * x
            dissipated += (c * xdot * xdot) * dt

            w.writerow([t, phase, target, x, xdot, Fn, E_store, dissipated])
            return x, xdot, Fn, E_store, dissipated

        print_every = max(1, int(PRINT_EVERY_S / dt))

        def run_with_viewer(loop_fn):
            with mujoco.viewer.launch_passive(model, data) as viewer:
                viewer.cam.distance = 0.25
                viewer.cam.lookat[:] = [0.0, 0.0, 0.015]
                viewer.cam.elevation = -30
                viewer.cam.azimuth = 90
                loop_fn(viewer)

        def experiment_loop(viewer=None):
            t = 0.0
            step = 0

            # --- Settle ---
            for _ in range(int(0.3 / dt)):
                step_once(q_start, t, "settle")
                t += dt
                step += 1
                if viewer:
                    viewer.sync()
                    time.sleep(dt)

            # --- Phase 1: Approach until contact ---
            n_approach = int(T_APPROACH / dt)
            approach_traj = ease_traj(q_start, MAX_APPROACH, n_approach)

            touch_q = None

            for q in approach_traj:
                x, xdot, Fn, E_store, E_diss = step_once(q, t, "approach")

                if step % print_every == 0:
                    print(f"t={t:6.3f}  q={q: .5f}  spike_x={x*1000: .3f}mm  Fn={Fn: .2f}N")

                # Detect first touch
                if touch_q is None and Fn >= FN_TOUCH_N:
                    touch_q = float(q)
                    print(f"\n✅ Contact detected at t={t:.3f}s, indenter_q={touch_q:.5f}, Fn={Fn:.2f}N\n")
                    break

                t += dt
                step += 1
                if viewer:
                    viewer.sync()
                    time.sleep(dt)

            if touch_q is None:
                print("\n❌ No contact detected during approach.")
                print("   Increase MAX_APPROACH or move indenter closer in the XML (indenter body pos z).\n")
                return

            # --- Phase 2: Compress extra 2mm after contact ---
            q_comp_end = min(touch_q + EXTRA_COMPRESS, MAX_APPROACH)
            n_comp = int(T_COMPRESS / dt)
            comp_traj = ease_traj(touch_q, q_comp_end, n_comp)

            for q in comp_traj:
                x, xdot, Fn, *_ = step_once(q, t, "compress")
                if step % print_every == 0:
                    print(f"t={t:6.3f}  q={q: .5f}  spike_x={x*1000: .3f}mm  Fn={Fn: .2f}N")
                t += dt
                step += 1
                if viewer:
                    viewer.sync()
                    time.sleep(dt)

            # --- Hold ---
            for _ in range(int(T_HOLD / dt)):
                x, xdot, Fn, *_ = step_once(q_comp_end, t, "hold")
                if step % print_every == 0:
                    print(f"t={t:6.3f}  q={q_comp_end: .5f}  spike_x={x*1000: .3f}mm  Fn={Fn: .2f}N")
                t += dt
                step += 1
                if viewer:
                    viewer.sync()
                    time.sleep(dt)

            # --- Release back up ---
            n_rel = int(T_RELEASE / dt)
            rel_traj = ease_traj(q_comp_end, q_start, n_rel)

            for q in rel_traj:
                x, xdot, Fn, *_ = step_once(q, t, "release")
                if step % print_every == 0:
                    print(f"t={t:6.3f}  q={q: .5f}  spike_x={x*1000: .3f}mm  Fn={Fn: .2f}N")
                t += dt
                step += 1
                if viewer:
                    viewer.sync()
                    time.sleep(dt)

            print(f"\nSaved log to: {OUT_CSV}")
            print("Done.\n")

        if view:
            run_with_viewer(experiment_loop)
        else:
            experiment_loop(None)

    print(f"Saved log to: {OUT_CSV}")


if __name__ == "__main__":
    main()

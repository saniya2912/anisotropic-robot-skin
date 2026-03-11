"""
visualize_cycle.py

Interactive MuJoCo viewer for the laminated cantilever spike simulation.
Runs one settle + load/unload cycle with the passive viewer open so you
can watch the beam deflect in real time.

Usage:
    python3 scripts/visualize_cycle.py [k_hinge]

    k_hinge  optional hinge stiffness, default 0.01
"""

import sys
import time
import re
import numpy as np
import mujoco
import mujoco.viewer

XML_PATH  = "models/leafspring_final.xml"

F_MAX     = 0.1   # N
T_SETTLE  = 0.5   # s
T_RAMP_UP = 1.0   # s
T_HOLD    = 0.5   # s
T_RAMP_DN = 1.0   # s

# Real-time playback speed (1.0 = wall-clock real time)
REALTIME_FACTOR = 0.5   # slow down to 0.5× so bending is easy to see


def load_model(k_hinge: float) -> mujoco.MjModel:
    with open(XML_PATH, "r") as f:
        xml = f.read()
    xml = re.sub(r'stiffness="[^"]*"', f'stiffness="{k_hinge:.8f}"', xml)
    return mujoco.MjModel.from_xml_string(xml)


def run(k_hinge: float = 0.01):
    model = load_model(k_hinge)
    data  = mujoco.MjData(model)
    dt    = model.opt.timestep

    tip_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "leaf1_tip")
    tip_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_site")

    mujoco.mj_forward(model, data)
    z0 = float(data.site_xpos[tip_site][2])

    total_time   = T_SETTLE + T_RAMP_UP + T_HOLD + T_RAMP_DN
    total_steps  = int(total_time / dt)
    sleep_per_step = dt / REALTIME_FACTOR

    print(f"k_hinge = {k_hinge}  |  total duration = {total_time:.1f} s  "
          f"({REALTIME_FACTOR}× speed → ~{total_time/REALTIME_FACTOR:.0f} s wall time)")
    print("Close the viewer window to exit early.\n")

    with mujoco.viewer.launch_passive(model, data) as v:

        # nice camera angle — side view to see bending
        v.cam.azimuth   = 90
        v.cam.elevation = -15
        v.cam.distance  = 0.12
        v.cam.lookat[:] = [0.01, 0.0, 0.03]

        for i in range(total_steps):
            if not v.is_running():
                break

            t = i * dt

            # ---- force profile ----
            if t < T_SETTLE:
                Fz    = 0.0
                phase = "settle  "
            elif t < T_SETTLE + T_RAMP_UP:
                s     = (t - T_SETTLE) / T_RAMP_UP
                Fz    = +F_MAX * s
                phase = "ramp-up "
            elif t < T_SETTLE + T_RAMP_UP + T_HOLD:
                Fz    = +F_MAX
                phase = "hold    "
            else:
                s     = (t - T_SETTLE - T_RAMP_UP - T_HOLD) / T_RAMP_DN
                Fz    = +F_MAX * (1.0 - min(s, 1.0))
                phase = "ramp-dn "

            data.xfrc_applied[:] = 0.0
            data.xfrc_applied[tip_body, 2] = Fz

            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)

            # terminal readout every 0.1 s
            if i % int(0.1 / dt) == 0:
                delta = z0 - float(data.site_xpos[tip_site][2])
                print(f"  t={t:5.2f}s  [{phase}]  Fz={Fz:+.3f} N  "
                      f"δ={delta*1e3:+6.3f} mm")

            v.sync()
            time.sleep(sleep_per_step)

    print("\nDone.")


if __name__ == "__main__":
    k = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
    run(k)

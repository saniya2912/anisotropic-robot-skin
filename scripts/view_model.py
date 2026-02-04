import time
import argparse
from pathlib import Path

import mujoco
import mujoco.viewer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    # ✅ Load plugins (elasticity lives here in your install)
    plugin_dir = Path(mujoco.__file__).resolve().parent / "plugin"
    mujoco.mj_loadAllPluginLibraries(str(plugin_dir))
    print("Loaded MuJoCo plugins from:", plugin_dir)

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()

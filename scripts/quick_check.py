import time
import mujoco
import mujoco.viewer

xml = """
<mujoco>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body pos="0 0 0.2">
      <freejoint/>
      <geom type="sphere" size="0.05"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)

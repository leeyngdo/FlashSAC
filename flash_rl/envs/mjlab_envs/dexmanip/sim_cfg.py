"""Simulation config for the DexManip motion-tracking env.

Declarative (isaaclab_envs style) port of DexManip's simulator builder:
  - builder:  src/envs/simulator/__init__.py  (build_sim / _build_mujoco / _build_nan_guard)
  - yaml:     config/envs/simulator/mujoco.yaml  (the ``sim:`` block, inlined below as literals)

``make_sim()`` reproduces the same ``mjlab.sim.SimulationCfg`` the Hydra+builder path produced,
with no yaml / hydra / OmegaConf dependency at runtime.
"""

from __future__ import annotations

from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.utils.nan_guard import NanGuardCfg


def make_sim() -> SimulationCfg:
    """Build the DexManip ``SimulationCfg`` from inlined ``mujoco.yaml`` values."""
    return SimulationCfg(
        nconmax=200,
        njmax=600,
        ls_parallel=True,
        contact_sensor_maxmatch=64,
        mujoco=MujocoCfg(
            timestep=0.002,
            integrator="implicitfast",
            impratio=1.0,
            cone="elliptic",
            jacobian="auto",
            solver="newton",
            iterations=20,
            tolerance=1.0e-8,
            ls_iterations=50,
            ls_tolerance=0.01,
            ccd_iterations=50,
            o_solref=None,
            o_solimp=(0.95, 0.99, 0.002, 0.5, 2.0),
            gravity=(0.0, 0.0, -9.81),
            disableflags=(),
            enableflags=(),
        ),
        nan_guard=NanGuardCfg(
            enabled=False,
            buffer_size=100,
            output_dir="/tmp/mjlab/nan_dumps",
            max_envs_to_dump=5,
        ),
    )

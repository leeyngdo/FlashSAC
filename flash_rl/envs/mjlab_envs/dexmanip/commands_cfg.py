"""Declarative command-manager cfg for the DexManip motion-tracking env.

Ports DexManip's ``config/envs/commands/motion_tracking.yaml`` together with
``src/envs/commands/__init__.py`` (``build_commands`` +
``_instantiate_dataclass``). The original builder reads the yaml's ``commands``
block, pops ``name`` / ``_target_``, and recursively instantiates the
``MotionTrackingCommandCfg`` dataclass (and its ``sampling`` / ``hand`` /
``object`` sub-dataclasses) from the remaining fields. Yaml-only fields such as
``input_dir`` that are not declared on the dataclass are silently dropped (they
are consumed by the scene/object builders instead).

Here those yaml literals are inlined as Python and the composition-derived
fields (robot body mapping / joint / site names, object entity names + mesh
paths/scales) are wired from ``robot_meta`` and ``object_info``. Returns
``{"motion": MotionTrackingCommandCfg(...)}`` to match mjlab's CommandManager
keying.
"""

from __future__ import annotations

from mjlab.managers.command_manager import CommandTermCfg

from flash_rl.envs.mjlab_envs.mdp import MotionTrackingCommandCfg
from flash_rl.envs.mjlab_envs.mdp.cmds.motion_tracking_cfg import (
    HandResetCfg,
    MotionSamplingCfg,
    ObjectCfg,
    ObjectSdfCfg,
    ObjectXfrcCfg,
    XfrcGainsCfg,
)


def make_commands(
    *,
    motion_file: str,
    input_dir: str,
    robot_meta: dict,
    object_info: dict,
) -> dict[str, CommandTermCfg]:
    """Build the motion-tracking command term dict.

    Args:
      motion_file: Path to the packed motion (``.pt``) / single ``.npz`` file.
        Forwarded verbatim to ``MotionTrackingCommandCfg.motion_file`` (matches
        the yaml ``motion_file: ???`` placeholder resolved by the caller).
      input_dir: Motion pool root. Consumed by the scene/object discovery; the
        command dataclass has no ``input_dir`` field, so it is accepted (to
        match the assembler contract) but not stored, mirroring DexManip's
        ``_instantiate_dataclass`` dropping it.
      robot_meta: Robot constants (``config/envs/robot/xhand_ghost_right.yaml``).
        Supplies ``entity_name``, ``body_mapping``, ``joint_names``,
        ``site_names`` and ``body_names['fingers']`` (-> ``finger_names``).
      object_info: Composition-derived per-side object metadata from
        ``scene_cfg.discover_objects``: dicts ``entity_names`` /
        ``mesh_paths`` / ``mesh_scales`` (side -> value).

    Returns:
      ``{"motion": MotionTrackingCommandCfg(...)}``.
    """
    del input_dir  # Not a dataclass field; consumed by scene/object builders.

    cfg = MotionTrackingCommandCfg(
        # --- args / composition-derived ---------------------------------
        motion_file=motion_file,
        entity_name=robot_meta["entity_name"],
        # Robot body <-> MANO joint mapping + name tables (from robot yaml).
        body_mapping=robot_meta["body_mapping"],
        finger_names=tuple(robot_meta["body_names"]["fingers"]),
        joint_names=robot_meta["joint_names"],
        site_names=robot_meta["site_names"],
        # --- mjlab CommandTermCfg fields (yaml literals) ----------------
        resampling_time_range=(1.0e9, 1.0e9),  # effectively no internal resample
        debug_vis=False,
        # --- reset-time motion-frame sampling ---------------------------
        sampling=MotionSamplingCfg(
            mode="uniform",
            kernel_size=1,
            lambda_=0.8,
            uniform_ratio=0.1,
            alpha=0.001,
        ),
        # --- hand: robot-side reset behaviour ---------------------------
        hand=HandResetCfg(
            joint_position_range=(0.0, 0.0),
            noise_to_initial_level=0.1,
            init_noise_scale={
                "wrist_trans": 0.01,  # m   (std on wrist translation)
                "wrist_rot_deg": 10.0,  # deg (std on wrist rotation)
                "finger_range_frac": 0.125,  # fraction of joint range (std on finger pos)
                "wrist_trans_vel": 0.01,  # m/s   (std on wrist translation velocity)
                "wrist_rot_vel": 0.01,  # rad/s (std on wrist rotation velocity)
                "finger_vel": 0.1,  # rad/s (REPLACES ref finger vel)
            },
        ),
        # --- object: SDF bake + pin (xfrc soft PD) ----------------------
        object=ObjectCfg(
            entity_names=object_info["entity_names"],
            mesh_paths=object_info["mesh_paths"],
            mesh_scales=object_info["mesh_scales"],
            sdf=ObjectSdfCfg(
                grid_extent=0.30,  # half-side of object-local SDF box (m)
                grid_n=48,  # voxels per axis (cube of N^3)
            ),
            pin_objects=True,
            pin_mode="xfrc",
            xfrc=ObjectXfrcCfg(
                kp=XfrcGainsCfg(pos=0.0, rot=0.0),
                kd=XfrcGainsCfg(pos=0.0, rot=0.0),
                omega_rot=20.0,  # >0 -> anisotropic inertia-tensor mode
                zeta_rot=1.0,
            ),
        ),
    )

    return {"motion": cfg}

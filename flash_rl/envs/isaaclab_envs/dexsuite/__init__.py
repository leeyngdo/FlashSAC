"""FlashSAC-local, state-based variant of the Isaac Lab dexsuite Kuka-Allegro reorient task.

The shipped ``Isaac-Dexsuite-Kuka-Allegro-Reorient-v0`` is a vision task: the *current object
position* is only observable through the point-cloud ``perception`` group (the ``policy`` group
carries object orientation + goal pose + last action, and ``proprio`` carries robot joint/contact
state). This variant turns it into a state-based task whose observation is aligned with the
NVIDIA IsaacGymEnvs ``AllegroKuka`` (reorientation) ``full_state`` layout used by the SAPG paper:

* adds privileged object **position** and **linear/angular velocity** (robot root frame) to the
  state (``policy``) group — AllegroKuka observes object pose + object lin/ang velocity explicitly;
* uses **single-frame** observations (``history_length = 1``) like AllegroKuka, which relies on the
  explicit velocities instead of frame stacking (the stock dexsuite groups stack 5 frames);
* drops the point-cloud ``perception`` group (never consumed by the state-based agent; removing it
  also avoids the per-step point-cloud computation and its Fabric visual warnings).

With ``obs_groups=["policy", "proprio"]`` this yields a ~166-dim state observation:
``policy`` = object_pos(3) + object_quat(4) + object_lin_vel(3) + object_ang_vel(3) + goal_pose(7)
+ last_action(23) = 43;  ``proprio`` = joint_pos(23) + joint_vel(23) + hand_tip_state(65) +
finger_contact(12) = 123.

Importing this module registers ``Isaac-Dexsuite-Kuka-Allegro-Reorient-State-v0``. The import is
triggered lazily by ``IsaacLabVectorEnv`` (via ``LOCAL_ISAACLAB_TASKS``) after the simulator app
starts, mirroring the local tracking tasks.
"""

import gymnasium as gym
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.dexsuite_kuka_allegro_env_cfg import (
    DexsuiteKukaAllegroReorientEnvCfg,
)
from isaaclab_tasks.manager_based.manipulation.dexsuite.mdp.observations import object_pos_b


def object_lin_vel_b(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object linear velocity (num_envs, 3) expressed in the robot root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w, obj.data.root_lin_vel_w)


def object_ang_vel_b(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object angular velocity (num_envs, 3) expressed in the robot root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w, obj.data.root_ang_vel_w)


@configclass
class DexsuiteKukaAllegroReorientStateEnvCfg(DexsuiteKukaAllegroReorientEnvCfg):
    """Kuka-Allegro reorient with an AllegroKuka-aligned, single-frame state observation."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # Privileged object pose + velocity (robot root frame), added next to object_quat_b so that
        # policy + proprio form a complete state observation without the point cloud.
        self.observations.policy.object_pos_b = ObsTerm(func=object_pos_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        self.observations.policy.object_lin_vel_b = ObsTerm(func=object_lin_vel_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        self.observations.policy.object_ang_vel_b = ObsTerm(func=object_ang_vel_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        # Single-frame observations (AllegroKuka full_state layout uses explicit velocities, not
        # frame stacking). The stock dexsuite groups use history_length=5.
        self.observations.policy.history_length = 1
        self.observations.proprio.history_length = 1
        # Drop the point-cloud perception group; this is a state-based setup and never consumes it.
        self.observations.perception = None
        # The ADR noise-curriculum terms that randomize the point-cloud observation now reference a
        # removed group, so disable them (object pose is observed as clean privileged state, which
        # also matches the AllegroKuka full_state layout). Other ADR terms (joint/object-quat noise,
        # gravity) are kept.
        if self.curriculum is not None:
            self.curriculum.object_obs_unoise_min_adr = None
            self.curriculum.object_obs_unoise_max_adr = None


gym.register(
    id="Isaac-Dexsuite-Kuka-Allegro-Reorient-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": DexsuiteKukaAllegroReorientStateEnvCfg},
)

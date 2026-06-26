"""Assembler for the modular motion-tracking environment configuration.

This module wires together the per-aspect configuration classes (scene, observations,
commands, rewards, terminations, events) that live in sibling modules into a single
:class:`TrackingEnvCfg`. It also defines the small action and curriculum configs that
do not warrant their own files. The simulation, decimation and episode settings in
``__post_init__`` mirror the upstream BeyondMimic ``tracking_env_cfg.py``.
"""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from .. import mdp as mdp
from .commands_cfg import CommandsCfg
from .events_cfg import EventCfg
from .observations_cfg import ObservationsCfg
from .rewards_cfg import RewardsCfg
from .scene_cfg import MySceneCfg
from .terminations_cfg import TerminationsCfg


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the motion-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        super().__post_init__()
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # viewer settings
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"

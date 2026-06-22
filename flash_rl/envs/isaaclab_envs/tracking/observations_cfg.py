"""Observation configuration for the G1 motion-tracking task.

Observation term functions are looked up from the :data:`...mdp.OBS_TERMS`
registry by name, so adding a new observation is a 3-step recipe: write the func
in ``mdp/obs/<file>.py`` -> add it to ``OBS_TERMS`` -> add an :class:`ObsTerm`
here referencing ``mdp.OBS_TERMS[...]``.

The privileged group is exposed under the name ``critic`` (not ``privileged``)
so FlashSAC auto-detects the asymmetric (actor/critic) observation split.
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from .. import mdp as mdp


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy (actor) group."""

        # observation terms (order preserved to match WBT)
        command = ObsTerm(func=mdp.OBS_TERMS["generated_commands"], params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(
            func=mdp.OBS_TERMS["motion_anchor_pos_b"],
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.25, n_max=0.25),
        )
        motion_anchor_ori_b = ObsTerm(
            func=mdp.OBS_TERMS["motion_anchor_ori_b"],
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        base_lin_vel = ObsTerm(func=mdp.OBS_TERMS["base_lin_vel"], noise=Unoise(n_min=-0.5, n_max=0.5))
        base_ang_vel = ObsTerm(func=mdp.OBS_TERMS["base_ang_vel"], noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.OBS_TERMS["joint_pos_rel"], noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.OBS_TERMS["joint_vel_rel"], noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.OBS_TERMS["last_action"])

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Privileged observations for the critic group (no noise corruption)."""

        # observation terms (order preserved to match WBT)
        command = ObsTerm(func=mdp.OBS_TERMS["generated_commands"], params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.OBS_TERMS["motion_anchor_pos_b"], params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.OBS_TERMS["motion_anchor_ori_b"], params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.OBS_TERMS["robot_body_pos_b"], params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.OBS_TERMS["robot_body_ori_b"], params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.OBS_TERMS["base_lin_vel"])
        base_ang_vel = ObsTerm(func=mdp.OBS_TERMS["base_ang_vel"])
        joint_pos = ObsTerm(func=mdp.OBS_TERMS["joint_pos_rel"])
        joint_vel = ObsTerm(func=mdp.OBS_TERMS["joint_vel_rel"])
        actions = ObsTerm(func=mdp.OBS_TERMS["last_action"])

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()

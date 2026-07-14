"""Event terms for the tracking task.

Aggregates the custom domain-randomization event functions used by the tracking
environment. Builtin event terms (e.g. ``randomize_rigid_body_material``,
``push_by_setting_velocity``) are provided by ``isaaclab.envs.mdp`` and resolved
through the aggregated ``mdp`` package.
"""

from .domain_rand import randomize_joint_default_pos, randomize_rigid_body_com

__all__ = ["randomize_joint_default_pos", "randomize_rigid_body_com"]

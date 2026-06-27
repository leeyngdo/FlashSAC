from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from mjlab.managers import CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

    from .motion_tracking import MotionTrackingCommand


@dataclass(kw_only=True)
class MotionSamplingCfg:
    """Reset-time motion-frame sampling parameters."""

    mode: Literal["adaptive", "uniform", "start"] = "uniform"
    start_frame: int = 0
    """For ``mode == "start"``: motion frame the rollout begins at, clamped
  per-trajectory to its last frame. 0 = original behaviour (frame 0). The
  recorded rollout then covers [start_frame, trajectory end]. Negative is a
  hard error. Ignored by the "uniform" and "adaptive" modes."""
    kernel_size: int = 1
    lambda_: float = 0.8
    """Kernel decay base for adaptive sampling."""
    uniform_ratio: float = 0.1
    alpha: float = 0.001


@dataclass(kw_only=True)
class HandResetCfg:
    """Robot-side reset behaviour."""

    joint_position_range: tuple[float, float] = (0.0, 0.0)
    noise_to_initial_level: float = 1.0
    """Global multiplier on all init-noise sigmas
  0.0 = deterministic ref pose (eval / rollout); 0.1 = ManipTrans default;
  1.0 = full per-component baseline. Effective sigma = baseline * multiplier.
  Eval-time swap target lives in cfg.eval.command.noise_to_initial_level."""
    init_noise_scale: dict[str, float] = field(
        default_factory=lambda: {
            "wrist_trans": 0.01,  # m   (baseline std on wrist translation)
            "wrist_rot_deg": 10.0,  # deg (baseline std on wrist rotation)
            "finger_range_frac": 0.125,  # fraction of joint range (baseline std on finger pos)
            "wrist_trans_vel": 0.01,  # m/s   (baseline std on wrist translation velocity)
            "wrist_rot_vel": 0.01,  # rad/s (baseline std on wrist rotation velocity)
            "finger_vel": 0.1,  # rad/s (baseline finger vel — REPLACES ref vel)
        }
    )
    """Per-component warm-start init-noise BASELINES (multiplied by
  ``noise_to_initial_level``). Set any baseline to 0 to disable that
  component independent of the global multiplier. ``finger_vel`` REPLACES
  the reference finger velocity rather than adding (matches ManipTrans
  behavior). Noise can't be an EventTermCfg — mjlab events fire before
  command_manager.reset, so they'd be overwritten by _resample_command's
  ref-pose write."""


@dataclass(kw_only=True)
class ObjectSdfCfg:
    """Per-object SDF grid bake parameters."""

    grid_extent: float = 0.30
    """Half-side of the object-local SDF box, in metres. Box is
  [-extent, extent]^3 (default 0.30 m → 0.60 m cube). Should comfortably
  enclose the object plus ~5 cm margin."""
    grid_n: int = 48
    """SDF grid resolution per axis (cube of N^3 voxels). Default 48 →
  ~6.25 mm voxels at default 0.30 m extent."""


@dataclass(kw_only=True)
class XfrcGainsCfg:
    """PD gain pair (linear + rotational)."""

    pos: float = 0.0
    rot: float = 0.0


@dataclass(kw_only=True)
class ObjectXfrcCfg:
    """xfrc soft-PD attractor gains (DexMachina-style)."""

    kp: XfrcGainsCfg = field(default_factory=XfrcGainsCfg)
    kd: XfrcGainsCfg = field(default_factory=XfrcGainsCfg)
    omega_rot: float = 0.0
    """If > 0, rotation PD switches to anisotropic inertia-tensor mode:
      τ = ω² · I_world · axis_angle(ref_q sim_q⁻¹)
        + 2 · ζ · ω · I_world · (ref_ang_vel - sim_ang_vel)
  Overrides kp.rot / kd.rot scalars."""
    zeta_rot: float = 1.0


@dataclass(kw_only=True)
class ObjectCfg:
    """Object interaction config — composition-derived metadata + SDF + pin."""

    # Composition-derived: filled by orchestrator, not yaml.
    entity_names: dict[str, str] | None = None
    """Side → scene entity name (e.g. {"right": "object_right"})."""
    mesh_paths: dict[str, str] | None = None
    """Side → absolute path to the visual mesh used for SDF bake. None disables
  baking."""
    mesh_scales: dict[str, float] | None = None
    """Side → mesh scale at SDF bake time. Defaults to 1.0 per side."""

    # Yaml-tunable.
    sdf: ObjectSdfCfg = field(default_factory=ObjectSdfCfg)
    pin_objects: bool = False
    pin_mode: Literal["xfrc", "none"] = "xfrc"
    """How to hold the object when ``pin_objects=True``.
  - "xfrc": soft PD on the freejoint via xfrc_applied. Decoupled gains via xfrc.kp/kd.
  - "none": no pinning. Off-switch."""
    xfrc: ObjectXfrcCfg = field(default_factory=ObjectXfrcCfg)


@dataclass(kw_only=True)
class MotionTrackingCommandCfg(CommandTermCfg):
    """Configuration for the hand motion tracking command."""

    motion_file: str
    entity_name: str
    finger_names: tuple[str, ...]
    joint_names: dict[str, list[str]]
    site_names: dict[str, dict]
    body_mapping: dict
    """Per-hand body-name mapping. Keys: 'all' (sequence of (body, mano_joint)
  pairs for all tracked non-tip bodies), 'level1' and 'level2' (dicts from
  finger name to the body used for the L1/L2 tracking reward). Supplied by
  the hand's asset_zoo constants module."""

    # Grouped sub-configs (yaml `commands.motion.{sampling,hand,object}`).
    sampling: MotionSamplingCfg = field(default_factory=MotionSamplingCfg)
    hand: HandResetCfg = field(default_factory=HandResetCfg)
    object: ObjectCfg = field(default_factory=ObjectCfg)

    # Debug-vis flags (require ``debug_vis=True`` on the parent CommandTermCfg).
    viz_robot_ghost: bool = False
    """When True (and ``debug_vis=True``), render the reference (target) hand
  as a translucent ghost overlay each frame. Used by the video script to draw
  the MANO reference alongside the simulated rollout."""
    viz_object_ghost: bool = False
    """When True (and ``debug_vis=True``), render the reference (target)
  object as a translucent ghost overlay. Independent of ``viz_robot_ghost`` so
  you can show only the hand reference, only the object reference, or both."""
    viz_human_keypoint: bool = False
    """When True (and ``debug_vis=True``), draw small spheres at every MANO
  keypoint the reward consumes (1 wrist + 5 tips + 12 non-tip joints per side),
  connected by green cylinders along each finger chain."""
    viz_object_collision: bool = False
    """When True, render the manipulated object as its CoACD/convex collision
  geoms instead of the smooth visual mesh. The convex pieces are authored into
  MuJoCo's collision render group; the render pass moves them into the visual
  group and colors them per-piece, and hides the visual geom. Consumed by
  ``src/utils/visualize.py``; unused by the record pass."""

    def build(self, env: ManagerBasedRlEnv) -> MotionTrackingCommand:
        from .motion_tracking import MotionTrackingCommand

        return MotionTrackingCommand(self, env)

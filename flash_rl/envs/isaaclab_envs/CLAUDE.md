# isaaclab_envs — agent guide

Vendored, KraftonLab-style modular IsaacLab G1 motion-tracking task. Math/logic copied
verbatim from `whole_body_tracking` (WBT); only layout / registries / reward grouping /
multi-dataset loader are new. See `README.md` for the user-facing guide; this file is the
working contract for editing the package.

## Hard rules

- **IsaacLab is NOT installed here.** Never import `isaaclab` from a tool/test you run. Verify
  every `.py` with `python -m py_compile <file>` only — it compiles files independently, so it
  will **not** catch cross-module relative-import errors. Trace those by hand.
- **No copyright headers** in this package (even though KraftonLab reference files have BSD-3
  headers — do not copy them).
- KraftonLab style: full type hints (modern `x | None`), Google-style docstrings, `snake_case`
  funcs, `PascalCase` classes, `SCREAMING_SNAKE` registries, `@configclass` with
  `__post_init__` calling `super().__post_init__()` **first**. `from __future__ import
  annotations` at the top of every term module so the `if TYPE_CHECKING: from isaaclab.envs
  import ManagerBasedRLEnv` guard works without importing IsaacLab at load.
- ruff/black line-length 120, double quotes.

## Import-light boundary (do not break)

- `__init__.py` (package root) is a bare docstring — no `isaaclab`, no `gym.register`.
- `utils/motion_loader.py` is **pure numpy + torch**. Never add an `isaaclab` import to it. It
  is the only import-light module besides the package root; tests import it directly.
- Everything under `tracking/` (and all `mdp/*/__init__.py` registry assemblies) imports
  IsaacLab transitively. Never import the `tracking` subtree from the package `__init__`.
- Registration is lazy: `make_isaaclab_env` (in `flash_rl/envs/isaaclab.py`) does
  `import flash_rl.envs.isaaclab_envs.tracking.config.g1` after `AppLauncher` and before
  `parse_env_cfg`.

## Layout / aggregation

`mdp/` is a **package-level** shared library (sibling of `tracking/`, mirroring KraftonLab's
`krafton_lab/mdp/`), NOT nested under the task — future envs reuse it. Per-aspect split of WBT's
single `tracking_env_cfg.py` lives under `tracking/`: `scene_cfg.py`, `observations_cfg.py`,
`rewards_cfg.py`, `terminations_cfg.py`, `events_cfg.py`, `commands_cfg.py`, plus the
`tracking_env_cfg.py` assembler. Each cfg file does `from .. import mdp as mdp` (two dots — `mdp`
is one level up, beside `tracking/`) and references `mdp.<symbol>` only.

`mdp/__init__.py` order is **load-bearing** — mirror WBT exactly:
`from isaaclab.envs.mdp import *` FIRST, then `from .cmds/.obs/.rews/.terms/.events import *`,
then explicit re-exports of `OBS_TERMS`, `REW_TERMS`, `REWARD_GROUPS`, `TERM_TERMS`. Local
star-imports must not shadow the builtins (`action_rate_l2`, `joint_pos_limits`,
`undesired_contacts`, `time_out`, `generated_commands`, etc.).

## Registries (functions, not classes)

| Registry | File | Type |
|---|---|---|
| `OBS_TERMS` | `mdp/obs/__init__.py` | `dict[str, Callable]` |
| `REW_TERMS` | `mdp/rews/__init__.py` | `dict[str, Callable]` |
| `TERM_TERMS` | `mdp/terms/__init__.py` | `dict[str, Callable]` |
| `REWARD_GROUPS` | `mdp/rews/__init__.py` | `dict[str, list[str]]` (group -> RewTerm field names) |

Values are term functions `func(env, **params) -> torch.Tensor`; cfg files build the
`*Term` wrapper. `re-export` modules (`obs/proprio.py`, `rews/regularization.py`,
`rews/safety.py`) import IsaacLab builtins and re-export them with `__all__` so
`from .module import *` works.

## Three naming spaces (do not conflate)

1. **Func name** = registry key = long WBT name, e.g. `motion_global_anchor_position_error_exp`.
2. **Cfg field name** = `RewTerm`/`ObsTerm`/`DoneTerm` attr = short flat name the override layer
   + YAML target, e.g. `motion_global_anchor_pos`, `motion_body_pos`, `anchor_pos`, `ee_body_pos`.
3. **`REWARD_GROUPS` entries** = cfg field names (not func names).

Indirections: termination `anchor_pos` -> `bad_anchor_pos_z_only`; `ee_body_pos` ->
`bad_motion_body_pos_z_only`. Obs fields `command`/`joint_pos`/`joint_vel`/`actions` ->
`generated_commands`/`joint_pos_rel`/`joint_vel_rel`/`last_action`; critic `body_pos`/`body_ori`
-> `robot_body_pos_b`/`robot_body_ori_b`.

## Add-a-term recipe (one registry + one cfg field)

- **Obs**: func in `mdp/obs/motion.py` (or re-export in `proprio.py`) → add to `OBS_TERMS` →
  add `ObsTerm` to `PolicyCfg`/`critic` in `observations_cfg.py`. Return `(num_envs, dim)`.
- **Reward**: func in `mdp/rews/{tracking,regularization,safety}.py` → add to `REW_TERMS` (+
  `REWARD_GROUPS` if grouped) → add `RewTerm` to the matching mixin in `rewards_cfg.py`. Return
  `(num_envs,)`, never `(num_envs, 1)`; exp tracking = `torch.exp(-error / std**2)`.
- **Termination**: func in `mdp/terms/tracking.py` → add to `TERM_TERMS` → add `DoneTerm` to
  `terminations_cfg.py`.

## Reward groups (multiple inheritance)

```python
@configclass
class RewardsCfg(TaskRewardsCfg, RegularizationRewardsCfg, SafetyRewardsCfg):
    pass
```

`@configclass`/dataclass field inheritance gathers all `RewTerm` fields from disjoint mixins
into one flat namespace; `RewardManager` iterates `cfg.__dict__.items()` → single summed scalar
reward (SAC-compatible). Mixins MUST have **disjoint** field names (else MRO shadows). The
override layer does `setattr(env_cfg.rewards, term, None)` to disable and
`env_cfg.rewards.<term>.weight` to rescale, so every term must stay a flat attr.

**Default weights = holosoma FastSAC preset, NOT raw WBT.** Use: `motion_global_anchor_pos`
1.0 (WBT 0.5), `motion_body_pos` 2.0 (WBT 1.0), `action_rate_l2` -1.0 (WBT -0.1). All `std`
params identical to WBT. `undesired_contacts` uses the **WBT 4-body** regex (`ankle_roll` +
`wrist_yaw`), not holosoma's 6-body. `feet_contact_time` exists in `rews/safety.py` but is
unused by default (matches WBT). `REWARD_GROUPS` is only for the override `_scale` + logging,
not multi-critic.

## Critic group naming (SAC parity)

The privileged obs group MUST be named `critic` (`ObservationsCfg.critic = PrivilegedCfg`), NOT
`privileged`: FlashSAC auto-detects asymmetry by checking for a `critic` key in
`single_observation_space` (`flash_rl/envs/isaaclab.py`). Set
`agent.asymmetric_observation=true` in the run script.

## Cross-module pitfalls (these break silently)

- `mdp/terms/tracking.py` imports the shared helper via `from ..rews.tracking import
  _get_body_indexes` (two dots up to `mdp`, into `rews.tracking`). Keep `_get_body_indexes`
  physically in `mdp/rews/tracking.py` — do not duplicate or move it.
- `mdp/obs/motion.py` imports `MotionCommand` via `from ..cmds.motion_command import
  MotionCommand` (relative, two dots).
- `mdp/cmds/motion_command.py` imports the loader via the **absolute** path
  `from flash_rl.envs.isaaclab_envs.utils.motion_loader import MotionLoader` (NOT relative —
  `utils` is a sibling of `tracking`, not under `mdp`).
- `mdp/events/domain_rand.py` keeps the IsaacLab framework-internal
  `from isaaclab.envs.mdp.events import _randomize_prop_by_op` verbatim — do not reimplement.
- `robots/g1.py`: `get_g1_cfg(...)` factory derives `stiffness = armature * nf**2`,
  `damping = 2 * dr * armature * nf` exactly as upstream; module-level `G1_CYLINDER_CFG =
  get_g1_cfg()` and `G1_ACTION_SCALE` keep DEFAULT numbers byte-identical to WBT. Actuator
  group keys are exactly 5: `legs`, `feet`, `waist`, `waist_yaw`, `arms`.
- `config/g1/__init__.py`: register `Isaac-Tracking-Flat-G1-v0` and
  `Isaac-Tracking-Flat-G1-WoSE-v0`. DROP `rsl_rl_cfg_entry_point`, the LowFreq variant, and the
  `LOW_FREQ_SCALE` import. Keep only `env_cfg_entry_point`.

## Motion loader

`MotionLoader(motion_files: str | list[str], body_indexes, device="cpu", balance_mode="frame")`
pools single path / list / directory-of-`.npz`. Records `clip_starts`, `clip_lengths`,
`time_step_total`. `balance_mode`: `"frame"` (uniform over frames) | `"motion"` (uniform over
clips then within). Helpers: `sample_start_frames`, `clip_id_of_frame`, `clip_end_of_frame`.
`.npz` keys: `fps`, `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`,
`body_ang_vel_w` (no separate root key). `MotionCommandCfg.motion_files` (MISSING; str wrapped
to list), `balance_mode`; preserves `anchor_body_name`/`body_names` + adaptive_* fields.

## Assets

G1 URDF at `assets/unitree_description/urdf/g1/main.urdf` (user pulls the Unitree description
tarball into `assets/`; `ASSET_DIR` resolves it). Motion `.npz` produced by WBT
`scripts/csv_to_npz.py` from retargeted `.csv`; place in `motions/`. See `README.md` for exact
commands.

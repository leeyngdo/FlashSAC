# FlashSAC IsaacLab Motion-Tracking Task (vendored, modular)

This package vendors the [BeyondMimic](https://beyondmimic.github.io/) G1 whole-body
motion-tracking task (from `whole_body_tracking`) into FlashSAC and reorganizes it in
**KraftonLab modular style**. The math/logic is copied verbatim from upstream; only the
module layout, registries, reward grouping, and multi-dataset motion loader are new.

It registers the gym ids `Isaac-Tracking-Flat-G1-v0` (state estimation) and
`Isaac-Tracking-Flat-G1-WoSE-v0` (without state estimation), wired into FlashSAC through
`flash_rl/envs/isaaclab.py` (`make_isaaclab_env` -> `gym.make` + `parse_env_cfg` on a
`ManagerBasedRLEnv`).

> The default reward weights are the **holosoma FastSAC preset**, not the raw WBT weights
> (see [Reward groups](#reward-groups)).

## Contents

- [Import-light contract](#import-light-contract)
- [Module layout](#module-layout)
- [Registries](#registries)
- [Adding a term (3-step recipe)](#adding-a-term-3-step-recipe)
- [Reward groups](#reward-groups)
- [Motion loader (multi-dataset + adaptive)](#motion-loader-multi-dataset--adaptive)
- [Three control layers](#three-control-layers)
- [Getting the G1 URDF + motion `.npz`](#getting-the-g1-urdf--motion-npz)
- [The `.npz` key schema](#the-npz-key-schema)

## Import-light contract

IsaacLab is a heavy, optional dependency. This package keeps a hard boundary so that pure
submodules stay importable (and unit-testable) **without IsaacLab installed**:

- `flash_rl/envs/isaaclab_envs/__init__.py` is a bare module docstring — **no** `isaaclab`
  imports, **no** `gym.register` at package import.
- Gym registration is triggered **explicitly and lazily** by `make_isaaclab_env`, which does
  `import flash_rl.envs.isaaclab_envs.tracking.config.g1` **after** `AppLauncher` starts the
  simulator (and before `parse_env_cfg`, which needs the task registered to resolve its
  `env_cfg_entry_point`).
- `utils/motion_loader.py` is **pure numpy + torch** — it never imports `isaaclab`. Tests
  import it directly (`from flash_rl.envs.isaaclab_envs.utils.motion_loader import MotionLoader`),
  not through the package root.
- The MDP term modules (`mdp/obs/*`, `mdp/rews/*`, `mdp/terms/*`, `mdp/cmds/*`,
  `mdp/events/*`) and their `__init__.py` registry assemblies **do** import `isaaclab`; they
  are not on the import-light path. Never import the `tracking` subtree from the package
  `__init__`.

Verification here is limited to `python -m py_compile <file>` (IsaacLab is not installed), so
cross-module relative imports must be traced by hand.

## Module layout

```
flash_rl/envs/isaaclab_envs/
  __init__.py                 # bare docstring (import-light)
  README.md, CLAUDE.md        # this guide + asset setup
  assets/__init__.py          # ASSET_DIR
  assets/unitree_description/ # user-placed G1 URDF (see below)
  motions/                    # user-placed .npz motion clips
  robots/{__init__, g1, actuator}.py   # G1 articulation factory + actuator helper
  utils/{__init__, motion_loader}.py   # pure numpy/torch multi-dataset loader
  mdp/                        # package-level shared MDP terms (KraftonLab-style; reusable across envs)
    __init__.py               # aggregator: isaaclab builtins first, then local re-exports
    cmds/{__init__, motion_command}.py
    obs/{__init__ (OBS_TERMS), motion, proprio}.py
    rews/{__init__ (REW_TERMS, REWARD_GROUPS), tracking, regularization, safety}.py
    terms/{__init__ (TERM_TERMS), tracking}.py
    events/{__init__, domain_rand}.py
  tracking/                   # the tracking task: per-aspect @configclass files + registration
    tracking_env_cfg.py       # TrackingEnvCfg assembler (ManagerBasedRLEnvCfg)
    scene_cfg.py              # MySceneCfg (terrain / robot=MISSING / lights / contacts)
    commands_cfg.py           # CommandsCfg (MotionCommandCfg defaults)
    observations_cfg.py       # ObservationsCfg: PolicyCfg + critic group, built from OBS_TERMS
    rewards_cfg.py            # reward group mixins + RewardsCfg + REWARD_GROUPS
    terminations_cfg.py       # TerminationsCfg
    events_cfg.py             # EventCfg
    config/
      g1/{__init__ (gym.register), flat_env_cfg}.py
```

`mdp/` is a **package-level** shared library of MDP terms (mirroring KraftonLab's `krafton_lab/mdp/`),
not nested under any single task — so future IsaacLab envs can reuse the same registries. The single
`tracking_env_cfg.py` from WBT is **split per MDP aspect**: one `@configclass` file each for scene /
observations / rewards / terminations / events / commands, plus a thin `tracking_env_cfg.py` assembler
that wires them as fields on `TrackingEnvCfg(ManagerBasedRLEnvCfg)`. Every cfg file imports
`from .. import mdp as mdp` and references everything as `mdp.<symbol>` — both IsaacLab builtins and
local custom funcs resolve through that single alias.

## Registries

Following the KraftonLab spirit, each MDP aspect exposes a `SCREAMING_SNAKE` registry assembled
in its `__init__.py`. Because the target is IsaacLab ManagerBased (not wrapper classes), the
registry **values are plain term functions** `func(env, **params) -> torch.Tensor`, and the cfg
files build the `RewTerm` / `ObsTerm` / `DoneTerm` wrappers around them.

| Registry | File | Type | Purpose |
|---|---|---|---|
| `OBS_TERMS` | `mdp/obs/__init__.py` | `dict[str, Callable]` | name -> observation func |
| `REW_TERMS` | `mdp/rews/__init__.py` | `dict[str, Callable]` | name -> reward func |
| `TERM_TERMS` | `mdp/terms/__init__.py` | `dict[str, Callable]` | name -> termination func |
| `REWARD_GROUPS` | `mdp/rews/__init__.py` | `dict[str, list[str]]` | group name -> list of RewTerm **field** names |

`mdp/__init__.py` aggregates in WBT's exact order: `from isaaclab.envs.mdp import *` **first**
(so builtins like `action_rate_l2`, `joint_pos_limits`, `undesired_contacts`, `time_out`,
`generated_commands`, `base_lin_vel`, `joint_pos_rel`, `randomize_rigid_body_material`,
`push_by_setting_velocity` resolve), then `from .cmds/.obs/.rews/.terms/.events import *`, then
explicit re-exports of the registry dicts.

### Func name vs cfg field name — keep them distinct

There are **three** naming spaces and they must stay consistent:

- **Func names** (registry keys in `REW_TERMS` / `OBS_TERMS` / `TERM_TERMS`): the long WBT
  function names, e.g. `motion_global_anchor_position_error_exp`.
- **Cfg field names** (the `RewTerm` / `ObsTerm` / `DoneTerm` attribute names on the cfg
  classes): the short flat names the override layer and YAML config target, e.g.
  `motion_global_anchor_pos`, `motion_body_pos`, `anchor_pos`, `ee_body_pos`.
- **`REWARD_GROUPS` list entries**: the **cfg field names** (not func names), used by the
  per-group `_scale` override and per-group logging.

Notable indirections to remember:

| Cfg field | Func used |
|---|---|
| `anchor_pos` (termination) | `bad_anchor_pos_z_only` (not `bad_anchor_pos`) |
| `ee_body_pos` (termination) | `bad_motion_body_pos_z_only` (not `bad_motion_body_pos`) |
| `command` (obs) | `generated_commands` |
| `joint_pos` / `joint_vel` (obs) | `joint_pos_rel` / `joint_vel_rel` |
| `actions` (obs) | `last_action` |
| `body_pos` / `body_ori` (critic obs) | `robot_body_pos_b` / `robot_body_ori_b` |

## Adding a term (3-step recipe)

This is the documented contract — add a term in exactly one registry + one cfg field.

**Add an observation**

1. Write the func in `mdp/obs/motion.py` (custom) or re-export a builtin in `mdp/obs/proprio.py`.
   Signature `func(env, command_name) -> torch.Tensor`; return shape `(num_envs, dim)`.
2. Add it to `OBS_TERMS` in `mdp/obs/__init__.py`.
3. Add an `ObsTerm(func=mdp.<func>, ...)` field to `PolicyCfg` or the `critic` group in
   `tracking/observations_cfg.py`.

**Add a reward**

1. Write the func in the right file under `mdp/rews/` (`tracking.py` / `regularization.py` /
   `safety.py`). Return shape `(num_envs,)` — **never** `(num_envs, 1)`. Exponential tracking
   is `torch.exp(-error / std**2)`.
2. Add it to `REW_TERMS` in `mdp/rews/__init__.py`, and add its cfg field name to the right
   list in `REWARD_GROUPS` if it belongs to a group.
3. Add a `RewTerm(func=mdp.<func>, weight=..., params={...})` field to the matching group mixin
   in `tracking/rewards_cfg.py` (`TaskRewardsCfg` / `RegularizationRewardsCfg` /
   `SafetyRewardsCfg`).

**Add a termination**

1. Write the func in `mdp/terms/tracking.py` (it may import the shared `_get_body_indexes` from
   `..rews.tracking`).
2. Add it to `TERM_TERMS` in `mdp/terms/__init__.py`.
3. Add a `DoneTerm(func=mdp.<func>, ...)` field to `tracking/terminations_cfg.py`.

## Reward groups

KraftonLab's true multi-critic uses a `RewardGroupManager` + nested `RewardGroupCfg`. The
FlashSAC adaptation is **ManagerBased-native**: define one `@configclass` mixin per group, then
combine them by **multiple inheritance**:

```python
@configclass
class RewardsCfg(TaskRewardsCfg, RegularizationRewardsCfg, SafetyRewardsCfg):
    pass
```

`@configclass` is built on dataclasses, so field inheritance gathers every `RewTerm` field from
all bases into one flat namespace. IsaacLab's `RewardManager` iterates `cfg.__dict__.items()`,
so all 9 terms are collected into a **single summed scalar reward** — compatible with FlashSAC
SAC. The three mixins have **disjoint field names** (no MRO collision), which is required.
`REWARD_GROUPS` is kept purely as a name->group mapping for the override layer's per-group
`_scale` and for per-group logging; it does **not** create separate critics.

Default weights (= holosoma `g1_29dof_wbt_fast_sac_reward` FastSAC preset; `std` params are
identical to WBT):

| Group | Field | Weight | `std` | WBT weight |
|---|---|---|---|---|
| task | `motion_global_anchor_pos` | 1.0 | 0.3 | 0.5 |
| task | `motion_global_anchor_ori` | 0.5 | 0.4 | 0.5 |
| task | `motion_body_pos` | 2.0 | 0.3 | 1.0 |
| task | `motion_body_ori` | 1.0 | 0.4 | 1.0 |
| task | `motion_body_lin_vel` | 1.0 | 1.0 | 1.0 |
| task | `motion_body_ang_vel` | 1.0 | 3.14 | 1.0 |
| regularization | `action_rate_l2` | -1.0 | — | -0.1 |
| regularization | `joint_limit` | -10.0 | — | -10.0 |
| safety | `undesired_contacts` | -0.1 | — | -0.1 |

The FastSAC delta vs raw WBT is exactly three weights: `action_rate_l2` -0.1 -> -1.0,
`motion_body_pos` 1.0 -> 2.0, `motion_global_anchor_pos` 0.5 -> 1.0.

`undesired_contacts` uses the WBT 4-body negative-lookahead regex (excludes
`left/right_ankle_roll_link` and `left/right_wrist_yaw_link`) — **not** the holosoma 6-body
regex, which is for a different contact-point naming.

`feet_contact_time` lives in `mdp/rews/safety.py` as an available helper but, matching WBT, is
**not** wired into any default `RewTerm`.

## Motion loader (multi-dataset + adaptive)

`utils/motion_loader.py` extends WBT's `MotionLoader` to pool multiple clips. It is pure
numpy/torch (no IsaacLab):

```python
MotionLoader(motion_files: str | list[str], body_indexes, device="cpu", balance_mode="frame")
```

- `motion_files` accepts a single `.npz` path, a list of paths, or a **directory** (globs
  `*.npz`). Each clip is loaded and frames are concatenated into pooled tensors, while
  per-clip boundaries are recorded as `clip_starts` (`long[num_clips]`) and `clip_lengths`
  (`long[num_clips]`); `time_step_total` is the total frame count.
- The property API (`joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`,
  `body_ang_vel_w`) is unchanged and indexes the pooled timeline with `self._body_indexes`
  selection — so `MotionCommand` consumes it exactly as before.
- `balance_mode`:
  - `"frame"` — uniform over **all** pooled frames.
  - `"motion"` — uniform over **clips**, then uniform within the chosen clip.
- Helpers: `sample_start_frames(n, generator=None, balance_mode=None) -> long[n]` (global start
  indices), `clip_id_of_frame(global_idx) -> long`, and `clip_end_of_frame(global_idx) -> long`
  (used to terminate/resample at a clip boundary so frames from different clips never blend).

`MotionCommand` (`mdp/cmds/motion_command.py`) keeps WBT's bin-based **adaptive sampling** but
builds the loader from `cfg.motion_files` + `cfg.balance_mode`, and respects `balance_mode` on
resample. `MotionCommandCfg` gains `motion_files: list[str] = MISSING` (a single str is
accepted and wrapped to `[motion_file]`) and `balance_mode: str = "frame"`, while preserving
`anchor_body_name` (`"torso_link"`) and the 14-body `body_names` list set in `flat_env_cfg.py`,
plus the adaptive fields `adaptive_kernel_size`, `adaptive_lambda`, `adaptive_uniform_ratio`,
`adaptive_alpha`. The number of bins is derived at runtime from `time_step_total`, not a cfg
field.

## Three control layers

You can control the task at three levels, applied in this order (later wins):

1. **Edit the vendored modular cfg** — change defaults directly in `tracking/*_cfg.py` (e.g.
   add a reward term, change the scene). This is the source-of-truth layer.
2. **Friendly grouped CLI blocks** — `env.reward.*`, `env.observation.*`,
   `env.termination.*`, `env.robot.*`, `env.pd_gain.*`, `env.motion.*` in
   `configs/env/isaaclab_tracking.yaml` / Hydra `--overrides`. These map onto the cfg through
   `apply_tracking_overrides` in `flash_rl/envs/isaaclab.py`. Reward overrides are grouped
   (`{group: {term: {weight?, std?, enabled?}, _scale?}}`); observation overrides target the
   **cfg field name** under `policy` / `critic` with `{enabled?, noise_scale?}`.
3. **`cfg_overrides` dot-path escape hatch** — `env.cfg_overrides` is a flat
   `{"dotted.path": value}` map walked by `_set_by_path` (supports dict members and attrs),
   applied **last** so dot-paths always win. Example:
   `{"sim.dt": 0.004, "scene.robot.actuators.legs.stiffness": 200.0}`.

`env.motion.motion_files` is **required** (a list of `.npz`, a directory of `.npz`, or a single
path). The 5 valid actuator group keys for `env.robot.groups` / `env.pd_gain.groups` are:
`legs`, `feet`, `waist`, `waist_yaw`, `arms` (each scales `stiffness` / `damping` /
`effort_limit_sim` / `armature` / `velocity_limit_sim`, dict-or-scalar per group).

For the full override contract and SAC parity defaults (`asymmetric_observation=true`,
`gamma=0.99`, `n_step=3`, 1024 envs), see `flash_rl/envs/isaaclab.py` and
`scripts/run_isaaclab_tracking.sh`.

## Getting the G1 URDF + motion `.npz`

These large binary assets are **not** vendored — you provide them.

### G1 robot description

The G1 articulation factory (`robots/g1.py`) expects the Unitree G1 description at:

```
flash_rl/envs/isaaclab_envs/assets/unitree_description/urdf/g1/main.urdf
```

Pull the description bundle used by BeyondMimic (from the upstream WBT instructions) and extract
it into `assets/`:

```bash
curl -L -o unitree_description.tar.gz \
  https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz
tar -xzf unitree_description.tar.gz -C flash_rl/envs/isaaclab_envs/assets/
rm unitree_description.tar.gz
```

`ASSET_DIR` (in `assets/__init__.py`) resolves this path for the robot cfg.

### Motion `.npz` clips

Reference motions are retargeted to G1 generalized coordinates (root pos[3] + root quat[4,
xyzw] + 29 joint angles) — e.g. the Unitree-retargeted
[LAFAN1 dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
(public mirror; follow the original licenses). Download G1 clips into `motions/lafan1_csv/`
(these `.csv` are gitignored — fetch locally):

```bash
BASE=https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/resolve/main/g1
mkdir -p flash_rl/envs/isaaclab_envs/motions/lafan1_csv
for f in walk1_subject1 dance1_subject2 run2_subject1; do
  curl -sL -o flash_rl/envs/isaaclab_envs/motions/lafan1_csv/$f.csv $BASE/$f.csv
done
```

Convert each retargeted `.csv` to the maximum-coordinates `.npz` (body pose / velocity via
forward kinematics) with the bundled **local** converter (wired to this task's `G1_CYLINDER_CFG`,
saves to disk, no WandB). **This is the one step that requires IsaacLab / Isaac Sim** — run it on
a machine with the `isaaclab` extra installed:

```bash
python scripts/csv_to_npz.py \
  --input_file flash_rl/envs/isaaclab_envs/motions/lafan1_csv/walk1_subject1.csv \
  --input_fps 30 --output_fps 50 --output_name walk1_subject1 --headless
# -> flash_rl/envs/isaaclab_envs/motions/walk1_subject1.npz
```

Then point `env.motion.motion_files` at the result (a single path, a list, or the directory —
the multi-dataset loader pools all clips it finds), e.g.
`--overrides 'env.motion.motion_files=[flash_rl/envs/isaaclab_envs/motions/walk1_subject1.npz]'`.

## The `.npz` key schema

Each `.npz` clip stores per-frame maximum-coordinates state (one row per output frame).
`MotionLoader` reads exactly these keys (no separate root key):

| Key | Shape | Meaning |
|---|---|---|
| `fps` | scalar / `[1]` | output frames per second |
| `joint_pos` | `[T, num_dofs]` | joint positions |
| `joint_vel` | `[T, num_dofs]` | joint velocities |
| `body_pos_w` | `[T, num_bodies, 3]` | body positions in world frame |
| `body_quat_w` | `[T, num_bodies, 4]` | body orientations (wxyz) in world frame |
| `body_lin_vel_w` | `[T, num_bodies, 3]` | body linear velocities in world frame |
| `body_ang_vel_w` | `[T, num_bodies, 3]` | body angular velocities in world frame |

`T` is the number of frames; `num_bodies` indexes the full robot body list, and
`MotionLoader` selects the tracked subset via `body_indexes` (derived from
`robot.find_bodies(body_names)` for the 14 tracked bodies).

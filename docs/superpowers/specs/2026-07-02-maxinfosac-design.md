# MaxInfoSAC integration into FlashSAC — design spec

Date: 2026-07-02
Branch: `feat/youngdo/maxinforl` (branched from `main` tip `f2da589`, per request; an
earlier draft on the diversity stack is preserved at `feat/youngdo/maxinforl-diversity-base`)
References: MaxInfoRL paper (arXiv 2412.12098), official torch reference
`sukhijab/maxinforl_torch` (cloned at /tmp/maxinforl_torch, commit HEAD of 2026-07-02).

## Goal

Add MaxInfoRL's directed-exploration mechanism (the MaxInfoSAC variant) to FlashSAC as a
config-gated feature. Default off; when disabled the agent byte-for-byte follows the
existing code path so concurrent experiments on other branches are unaffected.

## Algorithm summary (from the reference implementation)

MaxInfoSAC augments SAC's max-entropy objective with an information-gain intrinsic term
approximated by the disagreement of an ensemble of forward-dynamics models:

- **Ensemble**: E=5 independent MLPs (2 hidden layers × 256, ReLU) with fused heads,
  input `(s, a)`, output `{Δs = s' − s, r}` (reward head optional). Trained with MSE on
  normalized inputs/targets every update step, on replay `(s, a)` pairs. Adam, lr 3e-4,
  weight decay 0 (reference `examples/dmc/experiment.py`).
- **Info gain** `g(s,a)`: per output key, `log(EPS + Var_heads(pred)).mean(feature dims)`,
  averaged over keys, minus `log(EPS)` (the `learn_std=False` path in
  `MultiHeadGaussianEnsemble.get_disagreement` + `MaxInfoSAC.get_intrinsic_reward`);
  then standardized by a running "entropy normalizer".
- **Actor objective**: `E[α·logπ(a|s) − β·g(s, a_π) − Q(s, a_π)]` — the gradient of `g`
  flows through the sampled action into the frozen ensemble.
- **Critic target**: `r + γ(1−d)·(minQ'(s',a') − α·logπ(a'|s') + β·g(s',a'))`.
- **β auto-tuning**: keep a polyak copy of the actor (`actor_target`); update
  `log β` with loss `log β · (g(s, a_π) − g(s, a_π_target)).detach().mean()` — β rises
  when the current policy's info gain falls below the slower target policy's, keeping
  exploration pressure.
- **Normalizers**: running mean/std for ensemble input `(s,a)`, each output target, and
  the info-gain values. All Welford-style streaming stats.

## Mapping onto FlashSAC

FlashSAC specifics that shape the integration: categorical (distributional) double
critic trained by cross-entropy against a projected TD target; entropy enters the target
through the `actor_entropy` argument of `_compute_categorical_td_target`
(`target_bins = r + γ·(bins − actor_entropy)·(1−done)`); actor updates every
`actor_update_period` steps; everything torch.compile'd (CUDA graphs) with AMP.

1. **Actor loss** (`update_actor`): `(α·logπ − q).mean()` becomes
   `(α·logπ − β·g(s, a_π) − q).mean()`. `g` computed with ensemble grads disabled
   (same `requires_grad_(False)` discipline used for the critic in the actor step).
2. **Critic target** (`update_critic`): fold the bonus into the existing entropy slot:
   `actor_entropy := α·logπ(a'|s') − β·g(s', a')`. Subtracting `actor_entropy` in the
   categorical target then adds the info-gain bonus. Targets remain clamped to
   `[min_v, max_v]` as before.
3. **β (dyn scale)**: reuse `FlashSACTemperature` (a log-parameter scalar) as the β
   module; new `update_dyn_scale` mirrors `update_temperature`'s value-form loss
   `β·(g_rows − g_target_rows).mean()` (FlashSAC's temperature idiom; the reference uses
   the log-form — same fixed point and sign, different step scaling). Updated only on
   actor steps, like temperature.
4. **actor_target**: `Network` EMA copy of the actor (`ema_source=actor`,
   `ema_tau=critic_target_update_tau`), EMA'd every update step alongside the target
   critic. Called with `training=True` like the target critic so its UnitBatchNorm
   normalizes with batch stats (EMA covers parameters only, not running buffers).
5. **Ensemble step**: one MSE gradient step per `agent.update()` on the sampled batch,
   after actor/critic updates (reference order). Input/output normalizers update from
   the batch at the start of `update()`; the info-gain normalizer updates only in the
   actor path (reference behavior).
6. **Precision**: ensemble forward, disagreement, and normalizer math run in float32
   with autocast disabled — head-variance in fp16 is precision-fragile.
7. **Compile**: ensemble, β, and actor_target wrapped in
   `Network(compile_network=cfg.use_compile, compile_mode="default")` — inductor
   without CUDA graphs. Letting these extra networks join the CUDA-graph pool
   alongside the actor/critic can corrupt cudagraph-trees pool accounting ("live
   storage data ptrs ... not accounted for" during the actor's warmup; observed
   once at 4096 envs on dexsuite, not deterministic — an identical rerun passed).
   Excluding the maxinfo networks from graph capture removes the interaction class
   entirely at negligible cost. CUDA-graph outputs are still `.clone()`d before
   crossing compiled-call boundaries (existing discipline).

## Assumptions (autonomous-run decisions)

- **Base commit**: `main` (`f2da589`), as requested. Two diversity-branch commits are
  cherry-picked: `d942dc7` (float32 reward-normalizer stats — without it main crashes
  on any float64-reward env (gymnasium MuJoCo) with `normalize_reward=true`, maxinfo on
  or off, in the compiled categorical TD target's `scatter_add_`; verified with a
  maxinfo-disabled dry run) and `87af110` (obs-group concat + dexsuite tasks in the
  IsaacLab wrapper — required to run the Kuka-Allegro dexsuite benchmark). The per-row
  entropy CUDA-graph fix is diversity-only and not needed here.
- **Observation for the dynamics model**: the critic (full/privileged) observation, both
  as input and as Δ target. Actor gradients flow only through actions, so asymmetric
  actor observations need no special handling.
- **Reward target**: `batch["reward"]` as seen by the critic (normalized when
  `normalize_reward=true`); the ensemble's output normalizer standardizes it anyway.
- **n_step > 1**: the buffer's `next_observation` is `s_{t+n}` and `reward` the n-step
  return, so the ensemble predicts n-step-ahead quantities. Disagreement remains an
  epistemic-uncertainty signal; documented limitation, no masking (matches reference,
  which also never masks episode boundaries).
- **Checkpointing**: ensemble (with normalizer buffers), β, and actor_target are saved
  and loaded alongside the other networks when maxinfo is enabled. Old checkpoints
  without these files are not loadable into a maxinfo-enabled agent.

## New config fields (`FlashSACConfig` + `configs/agent/flashSAC.yaml`, all defaulted)

```yaml
maxinfo_enabled: false          # master switch
maxinfo_num_heads: 5            # ensemble size E
maxinfo_hidden_dim: 256         # ensemble MLP width
maxinfo_num_hidden_layers: 2    # reference features=(256, 256)
maxinfo_learning_rate: 3e-4     # constant Adam lr for the ensemble
maxinfo_learn_reward: true      # include reward head in ensemble targets
maxinfo_dyn_scale_init: 1.0     # β initial value
maxinfo_dyn_scale_auto: true    # false → β fixed at init value
```

β's optimizer/scheduler mirror the temperature's (peak lr + warmup-cosine); the ensemble
uses a constant lr like the reference.

## Files

- `flash_rl/agents/flashSAC/maxinfo.py` (new): `RunningNormalizer` (nn.Module with
  buffers → checkpointable), `EnsembleDynamics` (fused-head ensemble via batched
  einsum linear, like `EnsembleUnitLinear`), info-gain computation, `MaxInfoModules`
  bundle + factory, `update_ensemble`, `update_dyn_scale`.
- `flash_rl/agents/flashSAC/update.py`: optional maxinfo arguments on `update_actor` /
  `update_critic`; `None` (default) leaves the existing path untouched.
- `flash_rl/agents/flashSAC/agent.py`: config fields, module init, `update()`
  orchestration, save/load, metrics (`maxinfo/ensemble_loss`, `maxinfo/dyn_scale`,
  `maxinfo/info_gain`, `maxinfo/target_info_gain`, `maxinfo/next_info_gain`).
- `configs/agent/flashSAC.yaml`: the fields above, disabled by default.
- `scripts/maxinfo/run_mujoco.sh`, `scripts/maxinfo/run_isaaclab_dexsuite.sh`: runners
  mirroring the existing script style with maxinfo enabled (the dexsuite runner works
  after cherry-picking the obs-group/dexsuite wrapper support, see below).
- `tests/unit/agents/test_maxinfo.py`.

## Testing

Unit (CPU, `use_compile=false`, `use_amp=false`):
1. `RunningNormalizer` streaming stats match `torch.mean/std` over the concatenated
   stream.
2. Ensemble: output shapes `(B, D, E)`; heads differ at init; MSE decreases on a toy
   regression; disagreement on far-from-training inputs exceeds disagreement on
   training-distribution inputs after fitting.
3. `update_dyn_scale` direction: `g > g_target` drives β down, `g < g_target` up.
4. Effective-entropy fold: info gain shifts the categorical TD target upward (bonus
   sign test).
5. Integration: `FlashSACAgent` with `maxinfo_enabled=true` on synthetic Box spaces
   runs `process_transition` + several `update()`s with finite losses and emits the
   maxinfo metric keys; existing suite stays green (default-off regression).

End-to-end: MuJoCo Hopper-v4 dry runs with tiny step counts — (a) CPU functional run,
(b) GPU run with `use_compile=true`, `use_amp=true` to exercise the CUDA-graph path
(idle GPU, `CUDA_VISIBLE_DEVICES` pinned away from other sessions' devices).

## Risks / notes

- The categorical critic's support is bounded (`±normalized_G_max`); a large β could
  push targets into the clamp. β is auto-tuned against a standardized info gain
  (≈ N(0,1)), so magnitudes stay comparable to the entropy term; `maxinfo/dyn_scale`
  and `maxinfo/next_info_gain` are logged to watch for saturation.
- Ensemble adds one fused-MLP forward+backward per update plus two disagreement
  forwards (actor path 2B rows, critic path B rows); expected overhead is small
  relative to the critic but will be measured in the GPU dry run.

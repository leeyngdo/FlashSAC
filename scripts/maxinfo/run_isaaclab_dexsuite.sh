#!/bin/bash -ex
# MaxInfoSAC on the tuned dexsuite Kuka-Allegro Reorient env, mirroring
# scripts/run_isaaclab_dexsuite.sh exactly except for the maxinfo switch and the
# W&B routing. Usage: run_isaaclab_dexsuite.sh [gpu] [seed]
export OMNI_KIT_ACCEPT_EULA=YES
export UV_PROJECT_ENVIRONMENT=.venv-isaaclab
CUDA_VISIBLE_DEVICES=${1:-0} uv run --no-sync python train.py \
  --config_name flashSAC_base \
  --overrides seed=${2:-0} \
  --overrides logger_type=wandb \
  --overrides project_name=FlashRL-dexsuite \
  --overrides group_name=kuka-allegro-reorient \
  --overrides exp_name=maxinfosac \
  `#=== Environment (GPU sim) ===#` \
  --overrides env=isaaclab_dexsuite \
  --overrides env.env_name=Isaac-Dexsuite-Kuka-Allegro-Reorient-v0 \
  --overrides num_env_steps=10_000_003_072 \
  --overrides num_train_envs=4096 \
  --overrides num_eval_envs=null \
  --overrides num_record_envs=null \
  --overrides num_eval_episodes=4096 \
  --overrides num_record_episodes=0 \
  `#=== Agent (GPU sim) ===#` \
  --overrides agent=flashSAC \
  --overrides agent.buffer_max_length=10_000_000 \
  --overrides agent.buffer_min_length=100_000 \
  --overrides agent.buffer_device_type='cuda' \
  --overrides agent.sample_batch_size=8192 \
  --overrides agent.use_amp=true \
  --overrides updates_per_interaction_step=2 \
  `# 'auto' (max-autotune cudagraphs) crashes on this env at the first eval` \
  --overrides agent.compile_mode=default \
  `#=== MaxInfoSAC ===#` \
  --overrides agent.maxinfo_enabled=true \
  `#=== Benchmark default ===#` \
  --overrides agent.asymmetric_observation=false \
  --overrides gamma=0.99 \
  --overrides n_step=3 \
  `#=== Checkpointing ===#` \
  --overrides save_checkpoint_per_interaction_step=244_140

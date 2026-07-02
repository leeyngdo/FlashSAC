#!/bin/bash -ex
# MaxInfoSAC on Kuka-Allegro Reorient. Agent/training block matches the IsaacLab
# benchmark defaults (scripts/run_isaaclab.sh); env count and step budget follow the
# dexsuite gate setup. Usage: run_isaaclab_dexsuite.sh [gpu] [seed]
export OMNI_KIT_ACCEPT_EULA=YES
export UV_PROJECT_ENVIRONMENT=.venv-isaaclab
CUDA_VISIBLE_DEVICES=${1:-0} uv run --no-sync python train.py \
  --overrides env=isaaclab \
  --overrides env.env_name=Isaac-Dexsuite-Kuka-Allegro-Reorient-v0 \
  --overrides "env.obs_groups=[policy,proprio,perception]" \
  --overrides num_train_envs=4096 --overrides num_eval_envs=null --overrides num_record_envs=null \
  --overrides num_env_steps=30_000_000 \
  --overrides num_eval_episodes=4096 --overrides num_record_episodes=0 \
  --overrides agent.buffer_max_length=10_000_000 \
  --overrides agent.buffer_min_length=100_000 \
  --overrides agent.buffer_device_type='cuda' \
  --overrides agent.sample_batch_size=2048 \
  --overrides agent.use_amp=true \
  --overrides updates_per_interaction_step=2 \
  --overrides agent.asymmetric_observation=false \
  --overrides gamma=0.99 \
  --overrides n_step=3 \
  --overrides agent.maxinfo_enabled=true \
  --overrides logger_type=wandb --overrides project_name=FlashRL-maxinforl \
  --overrides group_name=kuka-allegro-reorient \
  --overrides exp_name=maxinfosac --overrides seed=${2:-0}

#!/bin/bash -ex
# MaxInfoSAC on Kuka-Allegro Reorient, mirroring the diversity gate-1 setup so the
# curves are directly comparable. Usage: run_isaaclab_dexsuite.sh [gpu] [seed]
export OMNI_KIT_ACCEPT_EULA=YES
export UV_PROJECT_ENVIRONMENT=.venv-isaaclab
CUDA_VISIBLE_DEVICES=${1:-0} uv run --no-sync python train.py \
  --overrides env=isaaclab \
  --overrides env.env_name=Isaac-Dexsuite-Kuka-Allegro-Reorient-v0 \
  --overrides "env.obs_groups=[policy,proprio,perception]" \
  --overrides num_train_envs=4096 --overrides num_eval_envs=null --overrides num_record_envs=null \
  --overrides num_env_steps=30_000_000 \
  --overrides num_eval_episodes=4096 --overrides num_record_episodes=0 \
  --overrides agent.maxinfo_enabled=true \
  --overrides logger_type=wandb --overrides project_name=FlashRL-sapg \
  --overrides group_name=maxinfo-gate1 \
  --overrides exp_name=maxinfosac --overrides seed=${2:-0}

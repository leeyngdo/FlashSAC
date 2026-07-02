#!/bin/bash
##################################################################################
# IsaacLab - Dexsuite Kuka-Allegro Reorient (GPU dexterous manipulation)
#
# Vanilla SAC baseline on the Kuka(iiwa7) + Allegro hand reorientation task.
# This is the Phase-0 baseline; SAPG (multi-policy) is layered on top via the flat
# `agent.sapg_*` overrides (e.g. agent.sapg_enabled=true agent.sapg_num_agents=4
# agent.sapg_target_entropy_grading=2.0 'agent.sapg_geom_alphas=[0,30,100,300]').
##################################################################################

env_name="Isaac-Dexsuite-Kuka-Allegro-Reorient-v0"
seeds=( 0 )
export WANDB_MODE=online

for seed in "${seeds[@]}"; do
    echo "$env_name, seed $seed"
    uv run python train.py \
        --config_name flashSAC_base \
        --overrides seed=${seed} \
        --overrides logger_type=wandb \
        --overrides project_name=FlashRL-sapg \
        `#=== Environment (GPU sim) ===#` \
        --overrides env=isaaclab_dexsuite_kuka \
        --overrides env.env_name=${env_name} \
        --overrides num_env_steps=100_000_000 \
        --overrides num_train_envs=1024 \
        --overrides num_eval_envs=null \
        --overrides num_record_envs=null \
        --overrides num_eval_episodes=1024 \
        --overrides num_record_episodes=0 \
        `#=== Agent (GPU sim) ===#` \
        --overrides agent=flashSAC \
        --overrides agent.buffer_max_length=10_000_000 \
        --overrides agent.buffer_min_length=100_000 \
        --overrides agent.buffer_device_type='cuda' \
        --overrides agent.sample_batch_size=2048 \
        --overrides agent.use_amp=true \
        --overrides updates_per_interaction_step=2 \
        `#=== Benchmark default ===#` \
        --overrides agent.asymmetric_observation=false \
        --overrides gamma=0.99 \
        --overrides n_step=3
done

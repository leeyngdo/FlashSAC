#!/bin/bash
##################################################################################
# IsaacLab SimToolReal Kuka-Sharpa with SAPG-style block-split exploration.
#
# Identical to run_isaaclab_simtoolreal.sh except for the SAPG block: 4 policy
# blocks (1024 envs each) sharing one backbone, actor conditioned on a per-block
# latent, critic block-unconditioned with leader-policy Bellman targets.
# Per-block target sigmas are a log-spaced grid (leader 0.1, followers hotter);
# reward-normalizer stats are anchored to the leader block. Eval uses the leader.
#
# Assets must be fetched once first: bash scripts/fetch_simtoolreal_assets.sh
##################################################################################

seeds=( 0 )

for seed in "${seeds[@]}"; do
    uv run python train.py \
        --config_name flashSAC_base \
        --overrides seed=${seed} \
        --overrides logger_type=wandb \
        --overrides project_name=FlashRL-simtoolreal \
        --overrides group_name=simtoolreal_sapg \
        --overrides exp_name=kuka_sharpa_sapg_m4 \
        `#=== Environment (GPU sim) ===#` \
        --overrides env=isaaclab_simtoolreal \
        --overrides env.env_name=Isaac-SimToolReal-Kuka-Sharpa-Direct-v0 \
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
        --overrides agent.compile_mode=default \
        `#=== SAPG (Split-only; see configs/agent/flashSAC.yaml) ===#` \
        --overrides agent.sapg_enabled=true \
        --overrides agent.sapg_num_agents=4 \
        --overrides "agent.sapg_target_sigmas=[0.1,0.15,0.25,0.4]" \
        --overrides agent.sapg_reward_norm_leader_only=true \
        `#=== Environment-specific ===#` \
        --overrides agent.asymmetric_observation=true \
        --overrides gamma=0.99 \
        --overrides n_step=3 \
        `#=== Checkpointing ===#` \
        --overrides save_checkpoint_per_interaction_step=244_140
done

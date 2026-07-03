#!/bin/bash
##################################################################################
# IsaacLab Dexsuite Kuka-Allegro Reorient (GPU Simulator)
#
# num_env_steps must be a multiple of num_train_envs:
#   10_000_003_072 = 4096 * 2_441_407 (~10G env steps, ~2.44M interaction steps)
# The default checkpoint interval saves only once at the very end of a run, so we
# save every 244_140 interaction steps (~1G env steps, 10 checkpoints total) instead.
##################################################################################

seeds=( 0 1000 2000 3000 4000 )

for seed in "${seeds[@]}"; do
    uv run python train.py \
        --config_name flashSAC_base \
        --overrides seed=${seed} \
        --overrides logger_type=wandb \
        --overrides project_name=FlashRL-dexsuite \
        --overrides group_name=dexsuite \
        --overrides exp_name=reorient \
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
        `#=== Benchmark default ===#` \
        --overrides agent.asymmetric_observation=false \
        --overrides gamma=0.99 \
        --overrides n_step=3 \
        `#=== Checkpointing ===#` \
        --overrides save_checkpoint_per_interaction_step=244_140
done

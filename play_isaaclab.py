import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import argparse
import random
import sys
from typing import MutableMapping, Optional

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from flash_rl.agents import create_agent
from flash_rl.envs.isaaclab import make_isaaclab_env
from flash_rl.envs.isaaclab_envs.utils.video import RESOLUTION_MAP, VideoRecorder
from flash_rl.types import Tensor


def play(args: argparse.Namespace) -> None:
    # Load config (same as train.py)
    OmegaConf.register_new_resolver("eval", lambda s: eval(s))
    hydra.initialize(version_base=None, config_path=args.config_path)
    cfg = hydra.compose(config_name=args.config_name, overrides=args.overrides)
    OmegaConf.resolve(cfg)

    # Seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    record = args.video
    # When recording we run headless with offscreen cameras enabled so the viewport renders without a GUI.
    env = make_isaaclab_env(
        env_name=cfg.env.env_name,
        num_envs=args.num_envs,
        seed=cfg.seed,
        headless=args.headless,
        enable_cameras=record,
        reward=cfg.env.get("reward", None),
        observation=cfg.env.get("observation", None),
        termination=cfg.env.get("termination", None),
        robot=cfg.env.get("robot", None),
        motion=cfg.env.get("motion", None),
        cfg_overrides=cfg.env.get("cfg_overrides", None),
        action_bound=cfg.env.get("action_bound", None),
    )

    # Create agent using config (same as train.py)
    _, env_info = env.reset(random_start_init=False)
    agent = create_agent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        env_info=env_info,
        cfg=cfg.agent,
    )
    agent.load(args.checkpoint_path)

    base_env = env.envs.unwrapped  # underlying IsaacLab ManagerBasedRLEnv (sim / scene / step_dt)

    recorder: Optional[VideoRecorder] = None
    if record:
        step_dt = float(base_env.step_dt)
        fps = int(max(1, min(round(1.0 / (step_dt * args.video_interval)), 60)))
        out_path = args.video_path or os.path.join(args.checkpoint_path, "play_video.mp4")
        recorder = VideoRecorder(out_path, resolution=RESOLUTION_MAP[args.resolution], fps=fps)
        recorder.initialize()
        # Position the persp camera to overview the scene.
        origins = base_env.scene.env_origins.cpu().numpy()
        center = origins.mean(axis=0)
        eye = center + (np.array(args.cam_eye) if args.cam_eye else np.array([5.0, 5.0, 4.0]))
        target = center + (np.array(args.cam_target) if args.cam_target else np.array([0.0, 0.0, 0.6]))
        base_env.sim.set_camera_view(eye=tuple(eye), target=tuple(target))
        # Ensure the reference-motion sphere markers are drawn into the offscreen recording.
        try:
            base_env.command_manager.set_debug_vis(True)
        except Exception as e:  # pragma: no cover - depends on live env
            print(f"[WARNING] could not enable command debug_vis: {e}")
        print(f"[INFO] Camera eye={eye.tolist()}, target={target.tolist()}; recording {args.video_length} steps")

    # Play loop
    observations, _ = env.reset(random_start_init=False)
    prev_transition: MutableMapping[str, Tensor] = {"next_observation": observations}
    completed_episodes = 0
    episode_returns = np.zeros(args.num_envs)
    step = 0

    while True:
        actions = agent.sample_actions(interaction_step=0, prev_transition=prev_transition, training=False)
        actions = np.array(actions)
        next_observations, rewards, terminateds, truncateds, infos = env.step(actions)
        step += 1

        if recorder is not None:
            base_env.sim.render()  # force a viewport render for headless capture
            if step % args.video_interval == 0:
                recorder.capture_frame()

        episode_returns += rewards
        episode_dones = np.logical_or(terminateds, truncateds)
        for idx in range(args.num_envs):
            if episode_dones[idx]:
                completed_episodes += 1
                print(f"Episode {completed_episodes}: return = {episode_returns[idx]:.2f}")
                episode_returns[idx] = 0.0

        observations = next_observations
        prev_transition = {"next_observation": observations}

        if recorder is not None:
            if step >= args.video_length:
                break
        elif completed_episodes >= args.num_episodes:
            break

    if recorder is not None:
        recorder.save()
    env.close()
    # IsaacSim's kit shutdown frequently hangs, leaving the process alive and holding GPU memory.
    # This is a one-shot play/record script, so hard-exit to guarantee the GPU is released.
    try:
        env.simulation_app.close()
    except Exception:
        pass
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play or record a trained FlashSAC agent in IsaacLab")
    parser.add_argument("--config_path", type=str, default="./configs")
    parser.add_argument("--config_name", type=str, default="flashSAC_base")
    parser.add_argument("--overrides", action="append", default=[])
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to agent checkpoint directory")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments")
    parser.add_argument("--num_episodes", type=int, default=10, help="Episodes to play (ignored when --video)")
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run headless (default). Use --no-headless for a GUI viewport (needs a display).",
    )
    parser.add_argument("--video", action="store_true", help="Record an mp4 via headless offscreen rendering.")
    parser.add_argument("--video_length", type=int, default=500, help="Number of sim steps to record.")
    parser.add_argument("--video_interval", type=int, default=1, help="Capture a frame every N steps.")
    parser.add_argument(
        "--video_path", type=str, default=None, help="Output mp4 path (default: <checkpoint_path>/play_video.mp4)."
    )
    parser.add_argument("--resolution", type=str, default="720p", choices=["1080p", "720p", "480p"])
    parser.add_argument(
        "--cam_eye", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"), help="Camera eye offset from center."
    )
    parser.add_argument(
        "--cam_target", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"), help="Camera target offset."
    )
    args = parser.parse_args()
    play(args)

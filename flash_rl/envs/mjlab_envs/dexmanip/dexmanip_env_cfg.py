"""DexManip env cfg bridge.

The default path intentionally delegates to the editable DexManip checkout
(``feat/youngdo/flashsac``) and its ``envs.make_env_cfg``. The local
``mjlab_envs/mdp`` implementation is kept as an opt-in native/override surface,
not as the source of truth for the environment.
"""

from __future__ import annotations

import json
import os
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

# DexManip defaults (config/envs/motion_tracking.yaml + commands object.density).
DECIMATION = 6
EPISODE_LENGTH_S = 20.0
COMMAND_NAME = "motion"
ENTITY_NAME = "robot"
_OBJECT_DENSITY = 800.0

# Package asset root (extracted motion/object data lives here; gitignored).
_ASSETS = Path(__file__).resolve().parents[1] / "assets"
_DEFAULT_MOTION = _ASSETS / "xhand/taco/right/20/motion.pt"


def _dexmanip_src_from_distribution() -> Path:
    """Return the source root for an editable-installed DexManip package."""
    try:
        dist = distribution("DexManip")
    except PackageNotFoundError as exc:
        raise FileNotFoundError(
            "DEXMANIP_STOPGAP requires DexManip to be installed in this env. "
            "Run `uv pip install --python $VENV --no-deps -e /path/to/DexManip`."
        ) from exc

    candidates: list[Path] = []
    direct_url = dist.read_text("direct_url.json")
    if direct_url:
        data = json.loads(direct_url)
        url = data.get("url")
        if isinstance(url, str) and url.startswith("file:"):
            candidates.append(Path(unquote(urlparse(url).path)))
    candidates.append(Path(dist.locate_file("")))

    for candidate in candidates:
        if (candidate / "src").exists() and (candidate / "config").exists():
            return candidate

    raise FileNotFoundError(
        "Installed DexManip distribution does not point at a source checkout "
        "with src/ and config/. Install it editable with `uv pip install --no-deps -e /path/to/DexManip`."
    )


def _resolve_dexmanip_src() -> Path:
    override = os.environ.get("DEXMANIP_SRC")
    if override:
        dex = Path(override)
        if not (dex / "src").exists():
            raise FileNotFoundError(f"$DEXMANIP_SRC does not contain src/: {dex}")
        return dex
    return _dexmanip_src_from_distribution()


def _default_viewer() -> Any:
    from mjlab.viewer import ViewerConfig

    return ViewerConfig(
        origin_type=ViewerConfig.OriginType.WORLD,
        distance=0.5,
        elevation=-20.0,
        azimuth=140.0,
        lookat=(0.0, 0.0, 0.0),
        width=640,
        height=480,
    )


def build_dexmanip_env_cfg(
    env_name: str,
    *,
    num_envs: int,
    seed: int,
    device: str = "cuda:0",
    motion: Any = None,
) -> Any:
    """Build a ManagerBasedRlEnvCfg for the dexmanip motion-tracking task.

    DexManip editable checkout by default; set ``DEXMANIP_NATIVE=1`` only when
    explicitly A/B testing the local native cfg assembly.
    """
    if os.environ.get("DEXMANIP_NATIVE"):
        return _build_native_env_cfg(env_name, num_envs=num_envs, seed=seed, device=device, motion=motion)
    return _build_upstream_env_cfg(env_name, num_envs=num_envs, seed=seed, device=device, motion=motion)


def _build_native_env_cfg(
    env_name: str, *, num_envs: int, seed: int, device: str = "cuda:0", motion: Any = None
) -> Any:
    """Self-contained native assembly from the owned mdp/ + dexmanip/ + robots/ modules.

    Mirrors DexManip ``make_env_cfg`` order, including the scene->command object
    coupling: objects are discovered from the motion.pt first, the resulting
    ``object_info`` feeds the motion command (mesh paths/scales), and the object
    entities go into the scene.
    """
    from mjlab.envs import ManagerBasedRlEnvCfg

    from ..robots.xhand import XHAND_META, XHAND_ROBOT_CFG
    from .actions_cfg import make_actions
    from .commands_cfg import make_commands
    from .curriculum_cfg import make_curriculum
    from .events_cfg import make_events
    from .observations_cfg import make_observations
    from .rewards_cfg import make_rewards
    from .scene_cfg import discover_objects, make_scene
    from .sim_cfg import make_sim
    from .terminations_cfg import make_terminations

    motion_file = str(motion) if motion else str(_DEFAULT_MOTION)
    input_dir = str(_ASSETS)

    object_entities, object_info = discover_objects(motion_file, input_dir, _OBJECT_DENSITY)
    scene = make_scene(
        num_envs=num_envs,
        robot_cfg=XHAND_ROBOT_CFG,
        robot_meta=XHAND_META,
        object_entities=object_entities,
    )
    commands = make_commands(
        motion_file=motion_file,
        input_dir=input_dir,
        robot_meta=XHAND_META,
        object_info=object_info,
    )
    return ManagerBasedRlEnvCfg(
        decimation=DECIMATION,
        episode_length_s=EPISODE_LENGTH_S,
        seed=seed,
        scene=scene,
        observations=make_observations(command_name=COMMAND_NAME, entity_name=ENTITY_NAME),
        actions=make_actions(robot_meta=XHAND_META),
        rewards=make_rewards(command_name=COMMAND_NAME, entity_name=ENTITY_NAME),
        commands=commands,
        terminations=make_terminations(command_name=COMMAND_NAME, entity_name=ENTITY_NAME),
        curriculum=make_curriculum(command_name=COMMAND_NAME, entity_name=ENTITY_NAME),
        events=make_events(),
        sim=make_sim(),
        viewer=_default_viewer(),
    )


def _build_upstream_env_cfg(
    env_name: str, *, num_envs: int, seed: int, device: str = "cuda:0", motion: Any = None
) -> Any:
    """Build via DexManip's original ``make_env_cfg`` ($DEXMANIP_SRC clone)."""
    del env_name  # DexManip's Hydra config owns the concrete env preset.
    import hydra
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    dex = _resolve_dexmanip_src()
    motion_file = str(motion) if motion else str(_DEFAULT_MOTION)
    robot_xml = dex / "assets/robot/xhand/right.xml"

    OmegaConf.register_new_resolver("eval", lambda s: eval(s), replace=True)
    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(version_base=None, config_dir=str(dex / "config")):
        cfg = hydra.compose(
            config_name="maniptrans",
            overrides=[
                f"num_envs={num_envs}",
                f"seed={seed}",
                f"device={device}",
                f"commands.motion_file={motion_file}",
                f"commands.input_dir={_ASSETS}",
                f"robot.xml_path={robot_xml}",
            ],
        )
    OmegaConf.resolve(cfg)

    try:
        from envs import make_env_cfg
    except ModuleNotFoundError as exc:
        if exc.name == "envs":
            raise ModuleNotFoundError(
                "DexManip is installed, but its `envs` package is not importable. "
                "Install the feat/youngdo/flashsac branch editable after its pyproject package list is updated."
            ) from exc
        raise

    return make_env_cfg(cfg)

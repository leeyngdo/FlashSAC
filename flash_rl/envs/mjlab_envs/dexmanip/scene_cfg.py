"""Scene wiring for the dexmanip motion-tracking task (terrain + objects + sensors).

Ports DexManip's ``src/envs/scene/__init__.py:build_scene`` together with its
helpers ``_object_setup.py`` (``discover_objects_from_motion`` + ``build_object``
via mjlab ``get_object_cfg``), ``_sensor_setup.py`` (``build_sensors``), and
``_terrain_setup.py`` (``build_terrain``). The declarative values transcribe
``config/envs/scene/base.yaml`` (plane terrain, sun light, groundplane
texture/material, ``env_spacing=0.5``, the two fingertip contact sensors) as
Python literals — no Hydra/OmegaConf at runtime.

Split from the original ``build_scene(cfg)`` into two assembler entry points:

  - ``discover_objects(motion_file, input_dir, density)`` torch.loads the motion
    ``.pt``, resolves per-side object mesh dirs, builds object ``EntityCfg`` via
    ``get_object_cfg``, and returns ``(object_entities, object_info)`` where
    ``object_info`` carries ``entity_names`` / ``mesh_paths`` / ``mesh_scales``
    (consumed by ``commands_cfg.make_commands``).
  - ``make_scene(num_envs, robot_cfg, robot_meta, object_entities)`` assembles the
    ``SceneCfg`` (terrain + robot + objects + sensors).
"""

from __future__ import annotations

from pathlib import Path

import torch
from mjlab.asset_zoo.objects.entity import get_object_cfg
from mjlab.entity import EntityCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import SensorCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.spec_config import LightCfg, MaterialCfg, TextureCfg

# --- config/envs/scene/base.yaml literals ------------------------------------

# scene.env_spacing  (tabletop scale; hand task)
ENV_SPACING = 0.5

# scene.terrain
TERRAIN_TYPE = "plane"  # "plane" | "generator"
TERRAIN_ENV_SPACING = 2.0


def _build_terrain() -> TerrainEntityCfg:
    """Mirror ``_terrain_setup.build_terrain`` for the inlined ``terrain`` block."""
    return TerrainEntityCfg(
        terrain_type=TERRAIN_TYPE,
        terrain_generator=None,
        env_spacing=TERRAIN_ENV_SPACING,
        max_init_terrain_level=None,
        lights=(
            LightCfg(
                name="sun",
                body="world",
                mode="fixed",
                type="directional",
                castshadow=True,
                pos=(0.0, 0.0, 1.5),
                dir=(0.0, 0.0, -1.0),
                cutoff=45.0,
                exponent=10.0,
            ),
        ),
        textures=(
            TextureCfg(
                name="groundplane",
                type="2d",
                builtin="checker",
                rgb1=(0.2, 0.3, 0.4),
                rgb2=(0.1, 0.2, 0.3),
                width=300,
                height=300,
                mark="edge",
                markrgb=(0.8, 0.8, 0.8),
            ),
        ),
        materials=(
            MaterialCfg(
                name="groundplane",
                rgba=(1.0, 1.0, 1.0, 1.0),
                texuniform=True,
                texrepeat=(4.0, 4.0),
                reflectance=0.2,
                texture="groundplane",
                geom_names_expr=("terrain$",),
            ),
        ),
    )


def _build_sensors(
    robot_entity_name: str, entities: dict[str, EntityCfg]
) -> tuple[SensorCfg, ...]:
    """Mirror ``_sensor_setup.build_sensors`` for the two inlined contact sensors.

    Both reference ``object_right`` on the secondary side; if that entity is
    absent (hand-only / partial-side mode), the sensor is silently dropped — same
    as the original ``needed - set(entities)`` guard.
    """
    r_fingertip_sites = (
        "contact_right_thumb_tip",
        "contact_right_index_tip",
        "contact_right_middle_tip",
        "contact_right_ring_tip",
        "contact_right_pinky_tip",
    )

    # (name, fields, reduce) -- the two sensors differ only in fields/reduce.
    specs = (
        ("r_fingertip_contact", ("found", "force"), "netforce"),
        ("r_fingertip_penetration", ("found", "dist"), "mindist"),
    )

    out: list[SensorCfg] = []
    for name, fields, reduce in specs:
        # Entities referenced by this sensor (primary -> robot, secondary -> obj).
        needed = {robot_entity_name, "object_right"}
        if needed - set(entities):
            continue  # silently skip; hand-only / partial-side mode
        out.append(
            ContactSensorCfg(
                name=name,
                primary=ContactMatch(
                    mode="site",
                    pattern=r_fingertip_sites,
                    entity=robot_entity_name,
                ),
                secondary=ContactMatch(
                    mode="body",
                    pattern="obj_right",
                    entity="object_right",
                ),
                fields=fields,
                reduce=reduce,
            )
        )
    return tuple(out)


def discover_objects(
    motion_file: str,
    input_dir: str,
    density: float,
) -> tuple[dict[str, EntityCfg], dict]:
    """Port of ``_object_setup.discover_objects_from_motion`` (+ ``build_object``).

    torch.loads the motion ``.pt`` (``weights_only=False``), resolves
    ``obj_dir = input_dir / pool_rel_dir / <side>_object_mesh_dir`` per side, and
    builds an object ``EntityCfg`` via mjlab ``get_object_cfg``.

    Returns ``(object_entities, object_info)`` where ``object_info`` has the
    per-side keys ``entity_names`` / ``mesh_paths`` / ``mesh_scales`` (the data
    the original builder wrote back onto ``motion_cfg.object`` for the command
    term; here it is returned for ``make_commands`` to consume).
    """
    object_info: dict = {
        "entity_names": {},
        "mesh_paths": {},
        "mesh_scales": {},
    }

    motion_file = str(motion_file)
    if not motion_file.endswith(".pt"):
        return {}, object_info

    packed = torch.load(motion_file, weights_only=False)
    pool_rel = packed.get("pool_rel_dir")
    if pool_rel is None:
        return {}, object_info

    pool_dir = Path(input_dir) / pool_rel
    density = float(density)
    sides = tuple(
        s for s in ("right", "left") if packed.get(f"{s}_object_mesh_dir") is not None
    )

    entities: dict[str, EntityCfg] = {}
    entity_names: dict[str, str] = {}
    mesh_paths: dict[str, str] = {}
    mesh_scales: dict[str, float] = {}

    for side in sides:
        mesh_rel = packed.get(f"{side}_object_mesh_dir")
        if mesh_rel is None:
            continue
        obj_dir = str(pool_dir / mesh_rel)
        scale = float(packed.get(f"{side}_object_mesh_scale", 1.0))
        entity_name = f"object_{side}"
        body_name = f"obj_{side}"

        entities[entity_name] = get_object_cfg(
            obj_dir=obj_dir,
            name=body_name,
            density=density,
            mesh_scale=scale,
        )
        entity_names[side] = entity_name
        # Motion command's SDF baker expects an explicit mesh file path
        # (legacy train_single_object_teacher.py appends "visual.obj").
        mesh_paths[side] = str(Path(obj_dir) / "visual.obj")
        mesh_scales[side] = scale

    object_info["entity_names"] = entity_names
    object_info["mesh_paths"] = mesh_paths
    object_info["mesh_scales"] = mesh_scales
    return entities, object_info


def make_scene(
    *,
    num_envs: int,
    robot_cfg: EntityCfg,
    robot_meta: dict,
    object_entities: dict[str, EntityCfg],
) -> SceneCfg:
    """Port of ``scene/__init__.py:build_scene``.

    Assembles ``SceneCfg`` from the robot entity (keyed by
    ``robot_meta['entity_name']``), the discovered object entities, the inlined
    plane terrain, and the inlined fingertip contact sensors.
    """
    robot_entity_name = robot_meta["entity_name"]
    entities: dict[str, EntityCfg] = {robot_entity_name: robot_cfg}
    entities.update(object_entities)

    return SceneCfg(
        num_envs=int(num_envs),
        env_spacing=float(ENV_SPACING),
        terrain=_build_terrain(),
        entities=entities,
        sensors=_build_sensors(robot_entity_name, entities),
    )

"""Object SDF (Signed Distance Field) baking and caching utilities.

Converts an object mesh into a 3D voxel grid of signed distances + gradients,
used by contact_match rewards and SDF-based observations.

Grid layout: (4, N, N, N) float32
  - channel 0: signed distance (positive outside, negative inside)
  - channels 1-3: analytic gradient (approx outward surface normal)

Grid spans [-extent, extent]^3 in the object-local frame.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import trimesh


# ---------------------------------------------------------------------------
# 1. Cache management
# ---------------------------------------------------------------------------


def _cache_path(
    mesh_path: str,
    extent: float,
    N: int,
    mesh_scale: float,
) -> Path:
    """Build the cache file path for a given mesh + bake parameters."""

    mesh_path_p = Path(mesh_path)
    cache_dir = mesh_path_p.parent / ".sdf_cache"
    key = hashlib.md5(
        f"{mesh_path_p.name}|method=v2_analytic|e={extent}|n={N}|s={mesh_scale}".encode()
    ).hexdigest()[:12]
    return cache_dir / f"sdf_v2_e{extent}_n{N}_s{mesh_scale}_{key}.pt"


def _load_cache(
    cache_file: Path,
    device: str,
) -> torch.Tensor | None:
    """Load cached SDF grid if it exists, else return None."""

    if cache_file.exists():
        return torch.load(cache_file, map_location=device, weights_only=True).to(
            device=device, dtype=torch.float32
        )
    return None


def _save_cache(
    grid: np.ndarray,
    cache_file: Path,
) -> None:
    """Save SDF grid to disk."""

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(grid), cache_file)


# ---------------------------------------------------------------------------
# 2. Mesh preprocessing
# ---------------------------------------------------------------------------


def _preprocess_mesh(
    mesh_path: str,
    mesh_scale: float,
    max_faces: int = 16000,
) -> trimesh.Trimesh:
    """Load mesh, apply scale, and decimate if too dense.

    SDF voxel cell size is ~12mm at default extent=0.30 / N=48;
    detail finer than that is wasted. High-poly inputs (e.g. TACO ~100k faces)
    blow up closest_point memory/time, so we cap to max_faces.
    """

    mesh = trimesh.load(Path(mesh_path), process=False, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(
            f"SDF bake expected a single trimesh.Trimesh, got {type(mesh)} from {mesh_path!r}"
        )
    if mesh_scale != 1.0:
        mesh.apply_scale(mesh_scale)

    if len(mesh.faces) > max_faces:
        decim = mesh.simplify_quadric_decimation(face_count=max_faces)
        if isinstance(decim, trimesh.Trimesh) and len(decim.faces) > 0:
            mesh = decim

    return mesh


# ---------------------------------------------------------------------------
# 3. SDF + gradient computation
# ---------------------------------------------------------------------------


def _build_object_sdf(
    mesh: trimesh.Trimesh,
    extent: float,
    N: int,
    batch_size: int = 5000,
) -> np.ndarray:
    """Build (4, N, N, N) SDF + gradient for an object mesh.

    Samples N^3 points in [-extent, extent]^3, queries the mesh for
    closest-point distance and inside/outside, then computes:
      - channel 0: signed distance (positive outside, negative inside)
      - channels 1-3: analytic gradient (surface normal direction)
    """

    # Generate 3D grid points
    xs = np.linspace(-extent, extent, N, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)

    # Batch-query mesh (caps memory at ~1 GB per batch)
    closest_chunks, dist_chunks, inside_chunks = [], [], []
    for b in range(0, pts.shape[0], batch_size):
        e = min(b + batch_size, pts.shape[0])
        cb, db, _ = trimesh.proximity.closest_point(mesh, pts[b:e])
        closest_chunks.append(cb)
        dist_chunks.append(db)
        inside_chunks.append(mesh.contains(pts[b:e]))

    closest = np.concatenate(closest_chunks, axis=0)
    dist = np.concatenate(dist_chunks, axis=0)
    inside = np.concatenate(inside_chunks, axis=0)

    # Signed distance: negative inside, positive outside
    sd = np.where(inside, -dist, dist).astype(np.float32)
    sdf = sd.reshape(N, N, N)

    # Gradient: sign(sd) * (pts - closest) / |pts - closest|
    sign = np.where(inside, -1.0, 1.0).astype(np.float32)
    disp = (pts - closest).astype(np.float32)
    unit = disp / np.maximum(np.linalg.norm(disp, axis=-1, keepdims=True), 1e-8)
    grad = (sign[:, None] * unit).astype(np.float32)
    grad = grad.reshape(N, N, N, 3).transpose(3, 0, 1, 2)

    return np.concatenate([sdf[None], grad], axis=0)  # (4, N, N, N)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def bake_object_sdf_grid(
    mesh_path: str,
    mesh_scale: float,
    extent: float,
    N: int,
    device: str,
) -> torch.Tensor:
    """Bake (or load cached) SDF + gradient grid for an object mesh.

    Args:
      mesh_path: Path to .obj mesh file.
      mesh_scale: Scale factor applied to the mesh.
      extent: Half-size of the grid cube (grid spans [-extent, extent]^3).
      N: Number of voxels per axis.
      device: Torch device string.

    Returns:
      Tensor of shape (4, N, N, N) on the given device.
    """

    cache_file = _cache_path(mesh_path, extent, N, mesh_scale)

    cached = _load_cache(cache_file, device)
    if cached is not None:
        return cached

    mesh = _preprocess_mesh(mesh_path, mesh_scale)
    grid = _build_object_sdf(mesh, extent, N)

    _save_cache(grid, cache_file)
    return torch.from_numpy(grid).to(device=device, dtype=torch.float32)

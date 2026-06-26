"""Headless viewport video recording helpers for IsaacLab play scripts."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

RESOLUTION_MAP = {"1080p": (1920, 1080), "720p": (1280, 720), "480p": (854, 480)}


class VideoRecorder:
    """Capture the Isaac Sim perspective viewport to an H.264 mp4."""

    def __init__(
        self,
        output_path: str,
        resolution: tuple[int, int] = (1280, 720),
        fps: int = 30,
        warmup_frames: int = 10,
    ):
        self.output_path = output_path
        self.resolution = resolution
        self.fps = fps
        self.frames: list[np.ndarray] = []
        self._annotator: Any = None
        self._render_product: Any = None
        self._initialized = False
        self._warmup_frames = warmup_frames
        self._frame_count = 0

    def initialize(self, cam_prim_path: str = "/OmniverseKit_Persp") -> None:
        import omni.replicator.core as rep

        self._render_product = rep.create.render_product(cam_prim_path, self.resolution)
        self._annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self._annotator.attach([self._render_product])
        self._initialized = True
        print(f"[INFO] VideoRecorder: {self.resolution[0]}x{self.resolution[1]} @ {self.fps} fps -> {self.output_path}")

    def capture_frame(self) -> bool:
        if not self._initialized:
            return False
        self._frame_count += 1
        if self._frame_count <= self._warmup_frames:
            return False
        try:
            rgb = self._annotator.get_data()
            if rgb is not None and getattr(rgb, "size", 0) > 0:
                frame = np.frombuffer(rgb, dtype=np.uint8).reshape(*rgb.shape)
                if frame.shape[2] >= 3:
                    self.frames.append(frame[:, :, :3].copy())
                    return True
        except Exception as exc:  # pragma: no cover - depends on live renderer
            print(f"[WARNING] capture_frame failed: {exc}")
        return False

    def save(self) -> None:
        if not self.frames:
            print("[WARNING] No frames captured; nothing to save.")
            return
        import imageio

        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        writer = imageio.get_writer(
            self.output_path,
            fps=self.fps,
            codec="libx264",
            pixelformat="yuv420p",
            output_params=["-crf", "23", "-preset", "medium", "-movflags", "+faststart"],
        )
        for frame in self.frames:
            writer.append_data(frame)
        writer.close()
        print(f"[INFO] Saved {len(self.frames)} frames ({len(self.frames) / self.fps:.1f}s) -> {self.output_path}")

    def __len__(self) -> int:
        return len(self.frames)

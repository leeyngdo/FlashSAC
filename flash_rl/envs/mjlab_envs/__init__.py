"""mjlab_envs — vendored, FlashSAC-owned mjlab environments.

Sibling of :mod:`flash_rl.envs.isaaclab_envs`, and the exact analogue of that
package: PURE CONTENT (cfg + MDP terms + override seam), no VectorEnv wrapper.
The wrapper and entry point live one level up in :mod:`flash_rl.envs.mjlab`
(``make_dexmanip_env`` + ``MjlabVectorEnv``), which reaches DOWN into this
package — top-down, just like ``isaaclab.py -> isaaclab_envs``.

Houses dexterous-manipulation tasks ported from DAVIAN-Robotics/DexManip,
restructured into the modular KraftonLab/isaaclab_envs layout so the MDP terms
are first-class, editable FlashSAC code rather than an opaque upstream blob:

    flash_rl/envs/mjlab.py    # <- the wrapper + make_dexmanip_env entry point (NOT here)
    mjlab_envs/
    ├── mdp/               # modular term library you own (cmds, obs, rews, terms, events, actions)
    ├── dexmanip/          # the task: declarative *_cfg.py composing mdp terms + overrides seam
    │                      #   exports build_dexmanip_env_cfg + apply_dexmanip_overrides (content API)
    ├── robots/            # robot definitions (xhand)
    ├── utils/             # heavy shared plumbing (motion lib, sdf, ...)
    └── assets/            # local DexManip asset cache (gitignored; fetch locally)

Track upstream by updating DexManip's ``feat/youngdo/flashsac`` branch from
``main`` and diffing its term changes back into this FlashSAC-owned port.

NOTE: importing this package does NOT import mjlab/torch — ``make_dexmanip_env``
in ``mjlab.py`` does all heavy imports lazily, so ``import flash_rl.envs`` stays
cheap in the base interpreter; mjlab only needs to exist in the env that runs
the DexManip task.
"""

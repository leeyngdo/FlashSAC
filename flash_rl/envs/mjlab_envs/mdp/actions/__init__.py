"""Action term.

Ported verbatim from DexManip ``src/envs/actions/maniptrans.py`` (only the
``..commands.motion_tracking`` import was rewired to ``..cmds.``). mjlab action
terms scale internally and clip to [-1, 1] — matching ``MjlabVectorEnv``'s
pass-through action contract. Wiring lives in ``dexmanip/actions_cfg.py``.
"""

from .maniptrans import ManipTransAction, ManipTransActionCfg

__all__ = ["ManipTransAction", "ManipTransActionCfg"]

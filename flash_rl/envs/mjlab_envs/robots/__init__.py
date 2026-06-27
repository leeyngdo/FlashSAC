"""Robot definitions.

PORT-FROM: DexManip ``assets/robot/xhand`` + ``src/envs/scene/_robot_setup.py``.

Status: TODO — define the xhand (5-finger, ghost-forearm) entity here.
``_robot_setup`` loads the MJCF via an UNANCHORED ``MjSpec.from_file`` on a
CWD-relative ``xml_path`` (``assets/robot/xhand/right.xml``); when ported, the
path MUST be absolutized against this package (``Path(__file__).parent``), else
the env silently builds object-free or fails to compile.
"""

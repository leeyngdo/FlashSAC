#!/bin/bash
##################################################################################
# Fetch the SimToolReal robot/table URDF assets (~16 MB, gitignored).
#
# Copies the Kuka iiwa14 + SHARPA description and the table URDF from
# tylerlum/simtoolreal into flash_rl/envs/isaaclab_envs/assets/simtoolreal_description/.
##################################################################################
set -euo pipefail

REPO_URL="https://github.com/tylerlum/simtoolreal"
DEST="$(cd "$(dirname "$0")/.." && pwd)/flash_rl/envs/isaaclab_envs/assets/simtoolreal_description"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

git clone --depth 1 "$REPO_URL" "$TMP_DIR/simtoolreal"

mkdir -p "$DEST/urdf"
cp -r "$TMP_DIR/simtoolreal/assets/urdf/kuka_sharpa_description" "$DEST/urdf/"
cp "$TMP_DIR/simtoolreal/assets/urdf/table_narrow.urdf" "$DEST/urdf/"
mkdir -p "$DEST/licenses"
cp "$TMP_DIR/simtoolreal/assets/licenses/kukaiiwa-LICENSE.txt" "$DEST/licenses/"

echo "SimToolReal assets fetched into $DEST"

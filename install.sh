#!/usr/bin/env bash
# Install the sound-reader skill for Claude Code and/or Codex by symlinking it into their skill folders.
#   ./install.sh            -> both (whichever of ~/.claude, ~/.codex exists)
#   ./install.sh claude     -> Claude Code only
#   ./install.sh codex      -> Codex only
#   ./install.sh --copy ... -> copy instead of symlink
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)/skills/sound-reader"
MODE=link
TARGETS=()
for a in "$@"; do
  case "$a" in
    --copy) MODE=copy ;;
    claude) TARGETS+=("${CLAUDE_CONFIG_DIR:-$HOME/.claude}/skills") ;;
    codex)  TARGETS+=("${CODEX_HOME:-$HOME/.codex}/skills") ;;
    *) echo "unknown arg: $a" >&2; exit 1 ;;
  esac
done
if [ ${#TARGETS[@]} -eq 0 ]; then
  [ -d "${CLAUDE_CONFIG_DIR:-$HOME/.claude}" ] && TARGETS+=("${CLAUDE_CONFIG_DIR:-$HOME/.claude}/skills")
  [ -d "${CODEX_HOME:-$HOME/.codex}" ] && TARGETS+=("${CODEX_HOME:-$HOME/.codex}/skills")
fi
[ ${#TARGETS[@]} -eq 0 ] && { echo "Neither ~/.claude nor ~/.codex found. Pass 'claude' or 'codex' explicitly." >&2; exit 1; }

for t in "${TARGETS[@]}"; do
  mkdir -p "$t"
  dest="$t/sound-reader"
  if [ -e "$dest" ] || [ -L "$dest" ]; then
    echo "skip: $dest already exists (remove it to reinstall)"; continue
  fi
  if [ "$MODE" = copy ]; then cp -r "$SRC" "$dest"; else ln -s "$SRC" "$dest"; fi
  echo "installed ($MODE): $dest"
done

python3 - <<'PY' || echo "!! Python deps missing:  python3 -m pip install -r $(dirname "$SRC")/../requirements.txt"
import numpy, scipy, matplotlib
print("python deps ok")
PY
command -v ffmpeg >/dev/null && echo "ffmpeg ok" || echo "note: ffmpeg not found - only WAV files will work (apt/brew/winget install ffmpeg)"

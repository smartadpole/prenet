#!/usr/bin/env bash
set -euo pipefail

OUT=".codex/AGENTS.md"
RULE_DIR=".cursor/rules"

if [[ ! -f "$OUT" ]]; then
  touch "$OUT"
  echo "[INFO] created $OUT"
fi

# Auto-discover all .mdc files in RULE_DIR (flat only) and keep a stable order.
mapfile -t FILES < <(find "$RULE_DIR" -maxdepth 1 -type f -name "*.mdc" -printf "%f\n" | sort)

{
  echo "# AGENTS.md"
  echo
  echo "> Auto-generated from .cursor/rules/*.mdc. Do not edit by hand."
  echo
  echo "## Global principles"
  echo "- Follow the rules below. If rules conflict, prefer later sections."
  echo "- Keep diffs minimal; do not touch unrelated files."
  echo
  for f in "${FILES[@]}"; do
    path="$RULE_DIR/$f"
    [[ -f "$path" ]] || continue
    echo
    echo "## Cursor rule: $f"
    echo
    # Strip optional frontmatter blocks (--- ... ---) if present.
    awk '
      BEGIN{in_fm=0}
      NR==1 && $0 ~ /^---[[:space:]]*$/ {in_fm=1; next}
      in_fm==1 && $0 ~ /^---[[:space:]]*$/ {in_fm=0; next}
      in_fm==1 {next}
      {print}
    ' "$path"
  done
} > "$OUT"

echo "[OK] wrote $OUT"

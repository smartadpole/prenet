#!/usr/bin/env bash
set -euo pipefail

OUT=".codex/AGENTS.md"
RULE_DIR=".cursor/rules"

if [[ ! -f "$OUT" ]]; then
  touch "$OUT"
  echo "[INFO] created $OUT"
fi

# 你想让 Codex 吃哪些规则：这里直接全吃（也可改成白名单）
# todo: smart 2026-01-28 15:54 - auto discover all .mdc files in RULE_DIR
FILES=(
  "auto-changelog.mdc"
  "clean-code.mdc"
  "coding-style.mdc"
  "doc.mdc"
  "file_head.mdc"
  "no-repeat.mdc"
  "timing.mdc"
  "ui-language.mdc"
  "version-control.mdc"
)

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
    # 去掉可能存在的 frontmatter（很多 .mdc 顶部会有 --- ... ---）
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

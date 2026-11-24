#!/usr/bin/env bash
set -u
SET=$1; K=$2; N=$3; STORE=$4; LOCAL=$5; B=$6
name=shard-$(printf %03d $K)
test -e "$STORE/$SET/$name/result.json" && grep -q '"status": "completed"' "$STORE/$SET/$name/result.json" && exit 0
mkdir -p "$STORE/$SET"
mkdir "$STORE/$SET/$name.claim" 2>/dev/null || exit 0
echo "$(hostname)" > "$STORE/$SET/$name.claim/host"
"$B/run-shard.sh" "$LOCAL" "$SET" "$K" "$N"
mkdir -p "$STORE/$SET"
rm -rf "$STORE/$SET/$name.partial"
cp -r "$LOCAL/$SET/$name" "$STORE/$SET/$name.partial"
for f in started console.log exit; do cp "$LOCAL/$SET/$name.$f" "$STORE/$SET/$name.$f"; done
if (cd "$LOCAL/$SET/$name" && sha256sum $(ls) ) | (cd "$STORE/$SET/$name.partial" && sha256sum -c --quiet -); then
  mv "$STORE/$SET/$name.partial" "$STORE/$SET/$name" && rm -rf "$LOCAL/$SET/$name"
else
  echo "$(date -u +%FT%TZ) copy check failed for $SET $name; local copy kept" >> "$STORE/queue.log"
fi

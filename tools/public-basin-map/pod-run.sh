#!/usr/bin/env bash
# CPU-pod driver for the basin map at H (docs/public-basin-map.md, amendment of 23 November).
# Shards run on the pod's local disk; each finished shard is copied to the network volume, every file's SHA-256 is
# checked there, and only then is the local copy removed. Order: M and S, then Z built from M's labels.
# Usage: pod-run.sh WORKERS
set -u
W=${1:-22}
export PY=/root/env/bin/python REPO=/root/shrimp-memory DATA=/workspace/data HF=/root/hf
B=/workspace/data/audits/public-basin-map
STORE=$B/pod
LOCAL=/root/run
mkdir -p "$STORE" "$LOCAL"
log() { echo "$(date -u +%FT%TZ) $*" >> "$STORE/queue.log"; }

cat > /root/one-shard.sh <<'EOF'
#!/usr/bin/env bash
set -u
SET=$1; K=$2; N=$3; STORE=$4; LOCAL=$5; B=$6
name=shard-$(printf %03d $K)
test -e "$STORE/$SET/$name/result.json" && grep -q '"status": "completed"' "$STORE/$SET/$name/result.json" && exit 0
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
EOF
chmod +x /root/one-shard.sh
shards() { echo $(( ($1 + 7) / 8 )); }
groups() { python3 -c "import json;print(json.load(open('$B/starts/starts-$1.json'))['groups'])"; }
NM=$(shards $(groups M)); NS=$(shards $(groups S))
log "start: M $NM shards, S $NS shards, $W workers"
{ for k in $(seq 0 $((NM-1))); do echo "M $k $NM"; done; for k in $(seq 0 $((NS-1))); do echo "S $k $NS"; done; } \
  | xargs -P "$W" -L 1 bash -c 'exec /root/one-shard.sh "$0" "$1" "$2" "'"$STORE"'" "'"$LOCAL"'" "'"$B"'"'
log "M and S finished"
cd "$REPO"
"$PY" tools/analyze_public_basin_map.py --starts-dir "$B/starts" --run-root "$STORE" \
  --reference "$DATA/audits/p6-recovery/20261025T221011Z-recovery/reference-cycles.npz" \
  --output "$STORE/receipt-M-S.json" >> "$STORE/queue.log" 2>&1 || { log "analysis of M and S failed"; exit 1; }
"$PY" tools/prepare_public_basin_map.py --plane "$B/starts/plane.npz" --zoom-from "$STORE/labels-M.npz" \
  --output-dir "$B/starts" >> "$STORE/queue.log" 2>&1
if [ -e "$B/starts/starts-Z.npz" ]; then
  NZ=$(shards $(groups Z)); log "start: Z $NZ shards"
  for k in $(seq 0 $((NZ-1))); do echo "Z $k $NZ"; done \
    | xargs -P "$W" -L 1 bash -c 'exec /root/one-shard.sh "$0" "$1" "$2" "'"$STORE"'" "'"$LOCAL"'" "'"$B"'"'
  "$PY" tools/analyze_public_basin_map.py --starts-dir "$B/starts" --run-root "$STORE" \
    --reference "$DATA/audits/p6-recovery/20261025T221011Z-recovery/reference-cycles.npz" \
    --output "$STORE/receipt-M-S-Z.json" >> "$STORE/queue.log" 2>&1
fi
log "QUEUE_DONE"

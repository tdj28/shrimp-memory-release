#!/usr/bin/env bash
# Helper-pod driver for the basin map at H (docs/public-basin-map.md, amendment of 23 November, 17:00 UTC).
# Works through M, then S, then Z, each in reverse shard order, claiming every shard on the shared volume before it
# runs (the first pod works forward), so no shard runs twice. Z is built by the first pod's driver from M's labels;
# this driver builds it only if that driver logged that it gave up.
# Usage: helper-run.sh WORKERS
set -u
W=${1:-22}
export PY=/root/env/bin/python REPO=/root/shrimp-memory DATA=/workspace/data HF=/root/hf
B=/workspace/data/audits/public-basin-map
STORE=$B/pod
LOCAL=/root/run
mkdir -p "$LOCAL"
log() { echo "$(date -u +%FT%TZ) $(hostname) $*" >> "$STORE/queue.log"; }
cp "$B/one-shard.sh" /root/one-shard.sh && chmod +x /root/one-shard.sh
shards() { echo $(( ($1 + 7) / 8 )); }
groups() { python3 -c "import json;print(json.load(open('$B/starts/starts-$1.json'))['groups'])"; }
done_count() { ls "$STORE/$1"/shard-*/result.json 2>/dev/null | wc -l; }
run_set() {
  local n=$(shards $(groups $1))
  for k in $(seq $((n-1)) -1 0); do echo "$1 $k $n"; done \
    | xargs -P "$W" -L 1 bash -c 'exec /root/one-shard.sh "$0" "$1" "$2" "'"$STORE"'" "'"$LOCAL"'" "'"$B"'"'
}
log "helper start, $W workers"
run_set M; log "helper: M list done"
run_set S; log "helper: S list done"
NM=$(shards $(groups M)); NS=$(shards $(groups S))
until [ "$(done_count M)" -ge "$NM" ] && [ "$(done_count S)" -ge "$NS" ]; do sleep 60; done
for i in $(seq 1 30); do
  [ -e "$B/starts/starts-Z.npz" ] && break
  grep -q "analysis of M and S failed" "$STORE/queue.log" && break
  sleep 20
done
if [ ! -e "$B/starts/starts-Z.npz" ] && mkdir "$B/starts/.z-lock" 2>/dev/null; then
  log "helper: building Z"
  cd "$REPO"
  "$PY" tools/analyze_public_basin_map.py --starts-dir "$B/starts" --run-root "$STORE" \
    --reference "$DATA/audits/p6-recovery/20261025T221011Z-recovery/reference-cycles.npz" \
    --output "$STORE/receipt-M-S.json" >> "$STORE/queue.log" 2>&1
  "$PY" tools/prepare_public_basin_map.py --plane "$B/starts/plane.npz" --zoom-from "$STORE/labels-M.npz" \
    --output-dir "$B/starts" >> "$STORE/queue.log" 2>&1
fi
until [ -e "$B/starts/starts-Z.json" ]; do sleep 30; done
sleep 10
run_set Z; log "helper: Z list done"
log "HELPER_DONE"

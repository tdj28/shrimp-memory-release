#!/usr/bin/env bash
# Follow-on driver for the first pod (docs/public-basin-map.md, amendment of 23 November, 18:20 UTC). pod-run.sh ends
# its M and S list while the helpers' last claimed shards still run, so its analysis stops at missing starts and it
# exits without Z. This waits for pod-run.sh to exit and for Z's starts (built by whichever driver gets there, by the
# fixed rule), then works through Z forward under the same atomic claims as the helpers.
# Usage: pod-z-after.sh PID_OF_POD_RUN WORKERS
set -u
P=$1; W=${2:-22}
export PY=/root/env/bin/python REPO=/root/shrimp-memory DATA=/workspace/data HF=/root/hf
B=/workspace/data/audits/public-basin-map
STORE=$B/pod
LOCAL=/root/run
log() { echo "$(date -u +%FT%TZ) $(hostname) $*" >> "$STORE/queue.log"; }
while kill -0 "$P" 2>/dev/null; do sleep 30; done
log "pod-z-after: pod-run.sh has exited; waiting for Z's starts"
until [ -e "$B/starts/starts-Z.json" ]; do sleep 30; done
sleep 10
n=$(( ($(python3 -c "import json;print(json.load(open('$B/starts/starts-Z.json'))['groups'])") + 7) / 8 ))
log "pod-z-after: Z $n shards, $W workers, forward"
for k in $(seq 0 $((n-1))); do echo "Z $k $n"; done \
  | xargs -P "$W" -L 1 bash -c 'exec /root/one-shard.sh "$0" "$1" "$2" "'"$STORE"'" "'"$LOCAL"'" "'"$B"'"'
log "pod-z-after: Z list done"
log "QUEUE_DONE"

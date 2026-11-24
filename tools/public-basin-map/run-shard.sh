#!/usr/bin/env bash
# One shard of one start set of the basin map at H (docs/public-basin-map.md).
# Usage: run-shard.sh OUT SET K SHARDS [--preflight-only]; env PY, REPO, DATA, HF override the host paths.
set -u
OUT=$1; SET=$2; K=$3; N=$4; EXTRA=${5:-}
PY=${PY:-/archive/shrimp-memory-data/envs/llm-cpu/bin/python}
REPO=${REPO:-/data2/shrimp-memory}
A=${DATA:-/archive/shrimp-memory-data}/audits
HF=${HF:-/data2/neural-shrimp-data/models/hf/hub}
STARTS=$A/public-basin-map/starts/starts-$SET.npz
SHA=$(sha256sum "$STARTS" | cut -d' ' -f1)
D="$OUT/$SET/shard-$(printf %03d $K)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$REPO"
test -e "$D/result.json" && grep -q '"status": "completed"' "$D/result.json" && exit 0
mkdir -p "$OUT/$SET"
rm -rf "$D"
date -u +%FT%TZ > "$D.started"
nice -n 19 "$PY" -u tools/run_public_basin_map.py --cache-dir "$HF" \
  --vectors ${DATA:-/archive/shrimp-memory-data}/inputs/p8-reference/vectors.npz \
  --references $A/p6-recovery/20261025T221011Z-recovery/reference-cycles.npz \
  --reference-manifest $A/p6-recovery/20261025T221011Z-recovery/reference-manifest.json \
  --source-hold-results $A/p6-followup/20261025T220342Z-followup/continued-hold/results.json \
  --recovery-results $A/p6-recovery/20261025T221011Z-recovery/run/results.json \
  --references-sha256 d96f923d537a6bcd6b86be7fe91f29a4cba6a46e5e036c8dd408817281395fe0 \
  --reference-manifest-sha256 17b8e826a63a01d442877b6cf837272f760f49f8f4ba26aa6192294f47b6853f \
  --recovery-results-sha256 5de613795c4bc10e0584072fd7cb12586de4db869286d15cbbdc7f57dab37ef5 \
  --starts "$STARTS" --starts-sha256 "$SHA" --start-set "$SET" \
  --shard "$K" --shards "$N" --output-dir "$D" $EXTRA > "$D.console.log" 2>&1
echo $? > "$D.exit"

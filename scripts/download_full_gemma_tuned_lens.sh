#!/usr/bin/env bash
set -euo pipefail

# Download the full 34-layer Gemma 3 4B IT tuned-lens probe set from GCS.
# The repository commits only layers 20/25/26 to keep clone size reasonable.

DEST="${1:-artifacts/lenses/gemma3_4b_it/full_probes}"
SRC="gs://pt-vs-it-results/tuned_lens_probes_v3/gemma3_4b/it"

mkdir -p "${DEST}"
gcloud storage cp "${SRC}/checkpoint.json" "${DEST}/"
gcloud storage cp "${SRC}/training_summary.json" "${DEST}/"
for layer in $(seq 0 33); do
  gcloud storage cp "${SRC}/probe_layer_${layer}.pt" "${DEST}/"
done

find "${DEST}" -type f -print0 | xargs -0 shasum -a 256 > "${DEST}/SHA256SUMS.txt"
echo "Downloaded full Gemma 3 4B IT tuned-lens probes to ${DEST}"

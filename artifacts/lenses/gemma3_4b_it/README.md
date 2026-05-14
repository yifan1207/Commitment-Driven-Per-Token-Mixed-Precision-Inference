# Gemma 3 4B IT Tuned-Lens Probe Artifacts

This directory contains selected tuned-lens probes for **Gemma 3 4B IT**.

The project default should still be the raw logit lens, because Gemma's raw lens is competitive or better in the late layers that are most useful for routing. These tuned probes are included for ablations at earlier or mid/late probe layers.

## Files

```text
metadata/checkpoint.json
metadata/training_summary.json
probes/probe_layer_20.pt
probes/probe_layer_25.pt
probes/probe_layer_26.pt
```

## Source

```text
gs://pt-vs-it-results/tuned_lens_probes_v3/gemma3_4b/it/
```

The full 34-layer probe set is intentionally not committed. Only layers 20, 25, and 26 are included because they are the useful tuned-lens ablation points for the commitment router.

## Recommended Use

- Use layer 20 for a more aggressive mid-stack tuned-lens router.
- Use layers 25 or 26 to compare tuned lens against the raw logit lens near the recommended deployment region.
- Use raw logit-lens probes for later layers such as 29-32.

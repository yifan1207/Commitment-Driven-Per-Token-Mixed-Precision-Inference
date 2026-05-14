# Gemma 3 4B IT Lens Agreement Plot

This plot compares per-layer agreement with the final-layer greedy argmax for:

- **Raw logit lens:** the model's original unembedding/readout applied to intermediate residual states.
- **Tuned lens:** trained per-layer probes from `tuned_lens_probes_v3`.

For the mixed-precision project, the important pattern is that the tuned lens is better in earlier layers, while the raw logit lens catches up and becomes stronger in the late layers. This makes the raw lens the clean MVP determinator, with tuned probes kept as an ablation.

## Files

```text
gemma3_4b_it_raw_vs_tuned_top1_agreement.png
gemma3_4b_it_raw_vs_tuned_top1_agreement.pdf
gemma3_4b_it_raw_vs_tuned_top1_agreement.csv
```

The CSV columns are:

```text
layer,raw_agree,tuned_agree
```

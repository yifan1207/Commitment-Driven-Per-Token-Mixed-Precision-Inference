# Commitment-Driven-Per-Token-Mixed-Precision-Inference

For greedy decoding, an LLM only needs the argmax of the final-layer logits; for sampling, the full distribution shape matters but the answer is often locked in well before the last layer. We will exploit this asymmetry by routing each token through one of two cheaper pathways instead of the full high-precision stack, decided on the fly from a mid-layer probe.

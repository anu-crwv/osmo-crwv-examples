# osmo-crwv-examples

Runnable example workloads for [NVIDIA Osmo](https://nvidia.github.io/OSMO/) on [CoreWeave](https://coreweave.com/). Each example is a self-contained Osmo workflow YAML you can submit with `osmo workflow submit <file> --pool default`.

## Examples

| Example | What it does | Time |
|---------|--------------|------|
| [`av-pipeline/`](./av-pipeline) | Three-stage AV workload: Cosmos-Predict2 scene generation → GR00T N1.6-3B finetune → IsaacLab Forklift navigation eval. Real NVIDIA stacks end-to-end on RTX PRO 6000 Blackwell. Full artifact lineage + side-by-side base-vs-finetuned comparison videos in W&B. | ~90 min full run; ~25 min eval-smoke iteration |

## Prerequisites (common to all examples)

- An Osmo deployment you can `osmo login` into, with `default` pool quota.
- A Weights & Biases account (most examples log artifacts + runs to `osmo-workflow` project by default).
- A Hugging Face account with whatever model access each example requires (the README in each example directory spells out which gated models to request access to).

## Quick start

```bash
git clone https://github.com/anu-wandb/osmo-crwv-examples.git
cd osmo-crwv-examples

# Pick an example and read its README, then submit:
osmo workflow submit av-pipeline/av-pipeline.yaml --pool default
osmo workflow query  av-pipeline-<N>
```

Each example's README documents one-time setup (credentials, HF gated access), submission command, what artifacts/runs you should see in W&B, and any cluster-specific gotchas already worked around in the YAML.

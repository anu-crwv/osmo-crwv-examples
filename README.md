<p align="center">
  <img src="./assets/nvidia-logo-horiz-wht-16x9.png#gh-dark-mode-only" width="600" alt="NVIDIA" style="vertical-align: middle;" />
  <img src="./assets/nvidia-logo-horiz-blk-16x9.png#gh-light-mode-only" width="600" alt="NVIDIA" style="vertical-align: middle;" />
  <img src="./assets/Endorsed_primary_goldwhite.png#gh-dark-mode-only" width="600" alt="Weights & Biases" style="vertical-align: middle;" />
  <img src="./assets/Endorsed_primary_goldblack.png#gh-light-mode-only" width="600" alt="Weights & Biases" style="vertical-align: middle;" />
</p>

# Physical AI Data Factory with NVIDIA Osmo on CoreWeave — a VLA Finetuning pipeline


A reference **Physical AI / robotics pipeline** that takes a real teleop forklift
dataset all the way to an in-simulation policy evaluation, running end-to-end on
**[NVIDIA Osmo](https://nvidia.github.io/OSMO/)** workflows over a **[CoreWeave](https://www.coreweave.com/)**
GPU cluster, with **[Weights & Biases](https://wandb.ai/site/experiment-tracking/)** as the experiment and artifact tracking system of record. 

This repo is organized as a **crawl → walk → run** progression so you can prove each
piece cheaply before spending a full training node:

<p align="center">
  <img src="./assets/Osmo-DAG.png" width="400" alt="Weights & Biases" />
</p>

<p align="center"><em>Osmo DAG showing our PAI Workflow</em></p>


<p align="center">
  <img src="./assets/W&B-linage-view.png" width="5000" alt="Weights & Biases" />
</p>

<p align="center"><em>Weights & Biases workspace tracking linage of all generated artifacts and associated workdlows</em></p>


| Level | What | Where | Cost |
|---|---|---|---|
| 🐛 **Crawl** | Smoke-test each stage in isolation | [`workloads/stage-level-smoke-test/`](workloads/stage-level-smoke-test/) | minutes (finetune smoke needs a node) |
| 🚶 **Walk** | Run the **whole pipeline** at tiny scale | `workloads/full-pipeline/full-pipeline.yaml --set max_steps=100 n_episodes=2` | ~½ hr |
| 🏃 **Run** | Run the **whole pipeline** at full scale | `workloads/full-pipeline/full-pipeline.yaml` (defaults) | hours, full 8-GPU node |

> 🤖 **Using an AI coding agent on this repo?** Point it at [`AGENTS.md`](AGENTS.md)
> — the same material plus the operating rules, Osmo schema gotchas, and the
> hard-won per-stage fixes it needs to change these workflows safely.

---

## Background: the three systems

### Osmo — the workflow orchestrator
[Osmo](https://nvidia.github.io/OSMO/) is NVIDIA's orchestrator for ML/robotics
workloads on a GPU cluster. You describe a **workflow** in a YAML file — a set of
**tasks** grouped into **groups**, each task pinned to a **resource** (GPU/CPU/mem
request) and a container **image**, running a shell **command**. Osmo schedules
the tasks onto a **pool**, runs the containers, enforces ordering between tasks
(`inputs:`), and injects secrets (`credentials:`). You drive it from the `osmo`
CLI: `validate` (free schema check) → `submit` → `query` / `logs`. Nothing in
this repo talks to the cluster directly; everything is an Osmo workflow YAML.

### CoreWeave — the GPU cloud underneath
The pool runs on a **CoreWeave** GPU cluster: pool **`default`**, platform
**`gpu`**, **16× NVIDIA RTX PRO 6000 Blackwell** GPUs (compute capability
**sm_120**) across **two 8-GPU nodes**. The Blackwell arch matters: CUDA wheels
must be built for sm_120 (the **cu128** PyTorch index) or you get `no kernel
image is available`; prebuilt flash-attn doesn't cover sm_120 (the stages fall
back to an SDPA path). The 8-GPU finetune needs a *whole* free node — check
`osmo resource list` before submitting.

### Weights & Biases — experiment and artifact tracker 
[W&B](https://wandb.ai) does double duty here. Beyond the usual metrics, videos,
and run history, **artifacts are how stages hand data to each other** — there is
no shared filesystem between Osmo tasks. Each stage *downloads* the previous
stage's `:latest` artifact and *logs* its own:

```
  av-forklift-v1  →  av-cosmos-cap-v1  →  groot-av-finetuned  →  eval summaries → comparison
   (dataset)          (captioned)          (model)               (per-policy)     (verdict)
```

Everything lands in W&B project **`osmo-workflow`** under your `WANDB_ENTITY`, so
following the artifact edges gives you full provenance: any eval traces back to
the exact captioned episodes the policy trained on. The `inputs:` edges in the
YAML only enforce *ordering* — the **data** moves through W&B.

> **Alternative — Osmo-native data passing (no W&B in the data path).**
> [`workloads/full-pipeline/full-pipeline-osmo-native.yaml`](workloads/full-pipeline/full-pipeline-osmo-native.yaml)
> is the **same DAG**, but hands data between stages over **Osmo's object storage
> (CoreWeave AI Object Storage / CAIOS)** instead of W&B artifacts: each producer
> writes to `{{output}}` and each consumer reads `{{input:N}}` — Osmo's `client`
> sidecar uploads on task-complete / downloads on task-start under
> `s3://<workflows-bucket>/osmo_workflow_data/<wf-id>/<task>/`. **W&B stays for
> experiment tracking only** (metrics, tables, videos, summaries); no dataset/model
> blobs round-trip through it. Validated end-to-end — all stages COMPLETED and every
> handoff (dataset → captioned dataset → 21 GB checkpoint → eval summaries) moved
> through object storage. Trade-off: the W&B artifact *lineage graph* isn't built
> (data lineage lives in Osmo instead). See the [walk/run](#-walk--run--the-full-pipeline) section.

---

## The pipeline stages

| # | Stage | Container | Does | Output artifact | View Workspace
|---|-------|-----------|------|-----------------|---------------|
| 1 | **dataset-preprocess** | `pytorch:25.05` | `tduggan93/forklift` (152 eps) → GR00T LeRobot v2.1: per-episode parquets, 448×448 mp4s, **9-D state / 3-D action `[steer, throttle, lift]`**, `modality.json` | `av-forklift-v1` | [Dataset Exploration](https://wandb.ai/wandb-smle/osmo-workflow?nw=efikm97ky5)
| 2 | **cosmos-cap** | `pytorch:25.05` | Cosmos3-Nano Reasoner (vLLM) captions each clip; per-episode language | `av-cosmos-cap-v1` | [Dataset Exploration](https://wandb.ai/wandb-smle/osmo-workflow?nw=efikm97ky5)
| 3 | **vla-finetune** | `pytorch:25.05` | GR00T **N1.6-3B** finetune, DDP across 8 GPUs; W&B run `groot-finetune` | `groot-av-finetuned` | [Training Results](https://wandb.ai/wandb-smle/osmo-workflow?nw=sld6hn8d3v)
| 4a/4b | **base-eval ∥ finetuned-eval** | `isaac-lab:2.3.2` | Isaac Sim forklift eval, **in parallel**: base `nvidia/GR00T-N1.6-3B` vs the finetuned policy. W&B runs `sim-eval-base-model` / `sim-eval-finetuned-model` | eval summaries | [Evaluatio Workspace](https://wandb.ai/wandb-smle/osmo-workflow?nw=e2ge6zxmu7n)
| 5 | **compare** | `pytorch:25.05` | reads both eval summaries, logs the delta + verdict | `comparison.json` | [Evaluatio Workspace](https://wandb.ai/wandb-smle/osmo-workflow?nw=e2ge6zxmu7n)

**The eval is honest about what the policy controls.** Drive (`throttle`/`steer`
→ root velocity) and fork-height (`lift` → fork joint) are **VLA-driven** — if the
policy doesn't drive in and lower the forks, nothing happens. Only **braking** (a
stop clamp at the pallet) and the **final lift** (the pallet rides the forks and
rises *by the policy's own fork height*) are assisted. Base GR00T (no forklift
skill) vs the finetuned policy separates cleanly on this.

---

## Prerequisites (one-time)

**1. Osmo CLI + a free node**
```bash
osmo pool list        # expect pool `default`, ONLINE
osmo resource list    # GPU used/total per node — stage 3 needs a whole free 8-GPU node
```

**2. Credentials** — every task pulls from one Osmo credential, `workflow-secrets`:
```bash
osmo credential set workflow-secrets --type GENERIC --payload \
  HF_TOKEN=hf_xxxx NGC_API_KEY=nvapi-xxxx WANDB_API_KEY=xxxx WANDB_ENTITY=your-entity
```
Tasks reference exactly the keys they need (there is **no "import all" shorthand**):
```yaml
credentials:
  workflow-secrets:
    WANDB_API_KEY: WANDB_API_KEY    # <credential-key>: <ENV_VAR_NAME>
    WANDB_ENTITY: WANDB_ENTITY
```
The W&B keys are load-bearing — artifacts *are* the inter-stage data path.

**3. Gated Hugging Face access** — `nvidia/Cosmos3-Nano` (stage 2) is **gated**;
your `HF_TOKEN` account must be granted access on the model page. If it can't
load, stage 2 **falls back to template captions** and still produces
`av-cosmos-cap-v1`, so the pipeline doesn't hard-fail. `tduggan93/forklift` and
`nvidia/GR00T-N1.6-3B` are not gated.

---

## 🐛 Crawl — smoke-test each stage

Each file in [`workloads/stage-level-smoke-test/`](workloads/stage-level-smoke-test/) is a **standalone, tiny** version of one stage —
the fastest way to prove that stage's code runs green before chaining anything.
They read the prior stage's `:latest` artifact from W&B and log their own, so you
can run them **in order** as an end-to-end plumbing check, or individually
against existing artifacts.

```bash
osmo workflow validate workloads/stage-level-smoke-test/dataset-preprocess.yaml --pool default   # always validate first (free)
osmo workflow submit   workloads/stage-level-smoke-test/dataset-preprocess.yaml --pool default    # → av-forklift-v1
osmo workflow submit   workloads/stage-level-smoke-test/cosmos-cap.yaml         --pool default    # → av-cosmos-cap-v1
osmo workflow submit   workloads/stage-level-smoke-test/finetune.yaml           --pool default    # 50 steps → groot-av-finetuned
osmo workflow submit   workloads/stage-level-smoke-test/eval.yaml               --pool default    # 1 episode
```

| File | Tests | Scale | GPUs |
|------|-------|-------|------|
| `workloads/stage-level-smoke-test/dataset-preprocess.yaml` | data conversion | full dataset (CPU, fast) | 0 |
| `workloads/stage-level-smoke-test/cosmos-cap.yaml` | Cosmos captioning + vLLM serve | full | 2 |
| `workloads/stage-level-smoke-test/finetune.yaml` | GR00T DDP training loop | **50 steps** | 8 |
| `workloads/stage-level-smoke-test/eval.yaml` | Isaac Sim + policy rollout | **1 episode** | 2 |

> Smoke tests share the production artifact namespace. `workloads/stage-level-smoke-test/finetune.yaml`
> produces a **throwaway ~50-step model** — re-run the full pipeline for a real
> one. Because they are single-task workflows, you can override scale at submit
> with `--set-env` (e.g. `--set-env MAX_STEPS=20`).

---

## 🚶 Walk / 🏃 Run — the full pipeline

Both use the **same file**, [`workloads/full-pipeline/full-pipeline.yaml`](workloads/full-pipeline/full-pipeline.yaml)
— one Osmo workflow with all stages and the **parallel base-vs-finetuned eval**.
Scale is parameterized with Osmo `--set` (templated `{{ max_steps }}` /
`{{ n_episodes }}`, defaulting to full scale in the file's `default-values:`).

```bash
osmo resource list                                              # confirm a free 8-GPU node

# 🚶 Walk — tiny end-to-end, proves the whole DAG hangs together cheaply
osmo workflow submit workloads/full-pipeline/full-pipeline.yaml --pool default \
  --set max_steps=100 n_episodes=2

# 🏃 Run — full scale (uses default-values: max_steps=6000, n_episodes=20)
osmo workflow submit workloads/full-pipeline/full-pipeline.yaml --pool default
```

**Data-handoff variant** — [`full-pipeline-osmo-native.yaml`](workloads/full-pipeline/full-pipeline-osmo-native.yaml)
is the same DAG and the same `--set` knobs, but passes data between stages via
**Osmo object storage** (`{{output}}`/`{{input:N}}`) instead of W&B artifacts; W&B
is tracking-only. Submit it exactly the same way:

```bash
osmo workflow submit workloads/full-pipeline/full-pipeline-osmo-native.yaml --pool default \
  --set max_steps=100 n_episodes=3
```

> **Why `--set`, not `--set-env`?** In a *multi-task* pipeline, `--set-env`
> reaches only the lead task — it will **not** change `MAX_STEPS` on the finetune
> task. `--set` does submit-time templating of `{{ field }}` and is reliable
> across all tasks. (`--set-env` *does* work for the single-task `workloads/stage-level-smoke-test/` files.)

**Scale knobs** (`--set <field>=<value>`):

| Field | Stage | Meaning | Default (run) |
|-------|-------|---------|---------------|
| `max_steps` | vla-finetune | training steps | `6000` |
| `n_episodes` | base/finetuned-eval | rollout episodes per policy | `20` |

> On training length: more is **not** always better here — a 20k-step run overfit
> (the policy stopped driving and only lifted), while ~4–6k both drove and lifted.
> `6000` is the default for that reason. See [`AGENTS.md`](AGENTS.md).

---

## Everyday Osmo commands

```bash
osmo workflow validate <file>.yaml --pool default     # free schema check — do this before every submit
osmo workflow submit   <file>.yaml --pool default
osmo workflow query    <workflow-id>                  # status (per-task)
osmo workflow logs     <workflow-id> --task <task> -n 200   # logs (no -f/follow; re-run or poll query)
osmo workflow logs     <workflow-id> --error
osmo workflow list     --pool default                 # recent runs
osmo workflow cancel   <workflow-id>
```

---

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| Finetune trains the wrong number of steps | Use `--set max_steps=N` on the pipeline, **not** `--set-env` (it doesn't reach the finetune task). Confirm with `--dry-run`. |
| Stage can't find its input artifact | It downloads `<entity>/osmo-workflow/<name>:latest`. Confirm the prior stage logged it and `WANDB_ENTITY` matches. |
| `Cannot access gated repo` / 403 at stage 2 | `HF_TOKEN` lacks access to `nvidia/Cosmos3-Nano`. Request it; stage 2 falls back to template captions meanwhile. |
| `CUDA error: no kernel image is available` | Wrong-arch torch wheel. This cluster is Blackwell **sm_120** — torch must come from the **cu128** index. The NGC image ships it. |
| Workflow stuck `QUEUED` | No node has enough free GPUs; the finetune (8 GPU) needs a whole free node. `osmo resource list`. |
| Sim hangs at exit / zombie processes | The eval explicitly tears down its gRPC sim server — preserve that teardown if you edit it. |
| Validation error on `credentials:` / `env:` | `env:` is not a valid task field (use `export` in the script). `credentials:` must be `{<cred>: {<key>: <ENV_VAR>}}`. |

---

## Repository map

```
osmo-crwv-examples/
├── README.md                                 ← you are here
├── AGENTS.md                                 ← operating guide for AI coding agents
└── workloads/
    ├── stage-level-smoke-test/               ← 🐛 smoke-test each stage (standalone, tiny)
    │   ├── dataset-preprocess.yaml
    │   ├── cosmos-cap.yaml
    │   ├── finetune.yaml                     (50 steps)
    │   └── eval.yaml                         (1 episode)
    └── full-pipeline/
        ├── full-pipeline.yaml                ← 🚶 walk (--set small) / 🏃 run (defaults) — data hands off via W&B artifacts
        └── full-pipeline-osmo-native.yaml    ← same DAG, data hands off via Osmo object storage ({{output}}/{{input:N}}); W&B = tracking only
```

The `workloads/stage-level-smoke-test/` files are the **source of truth per stage**; `workloads/full-pipeline/full-pipeline.yaml`
is assembled from them (with the eval fanned out into parallel base/finetuned
tasks + a compare). Change a stage in `workloads/stage-level-smoke-test/`, prove it, then refold it into the
pipeline. `full-pipeline-osmo-native.yaml` is the same DAG with the W&B-artifact
data handoffs swapped for Osmo object-storage (`{{output}}`/`{{input:N}}`) reads —
keep the two pipeline files in sync when you change a stage's logic.

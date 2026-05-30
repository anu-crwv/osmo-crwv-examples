# AV Pipeline — Cosmos → GR00T → IsaacLab on Osmo / CoreWeave

A real end-to-end autonomous-vehicle workload that runs three NVIDIA stacks back-to-back on an Osmo+CoreWeave pool of RTX PRO 6000 Blackwell GPUs:

```
┌───────────────┐    ┌───────────────────┐    ┌────────────────────┐
│ Stage 1       │───▶│ Stage 2           │───▶│ Stage 3            │
│ cosmos-gen    │    │ groot-finetune    │    │ av-eval            │
│               │    │                   │    │                    │
│ Cosmos-Predict2│    │ NVIDIA Isaac-GR00T│    │ Isaac Lab Forklift │
│ Text2World    │    │ N1.6-3B finetune  │    │ nav, base vs ft    │
│ 2B / Blackwell│    │ 8× sm_120, DDP    │    │ chase-cam videos   │
└───────────────┘    └───────────────────┘    └────────────────────┘
   av-cosmos-scenes  ─▶ groot-av-finetuned ─▶ policy-eval comparison
   (dataset artifact)   (model artifact)       (W&B run with videos)
```

The pipeline:

1. **Stage 1 — Cosmos scene generation.** Real Cosmos-Predict2 inference (text2image → video2world) on 8 forklift-warehouse prompts. Produces a LeRobot-v2 dataset of generated driving clips + per-frame trajectories. Logged as the W&B artifact `av-cosmos-scenes`.
2. **Stage 2 — GR00T N1.6-3B finetune.** Distributed (DDP, 8 GPUs) finetune on the Cosmos dataset using a `NEW_EMBODIMENT` vehicle modality config (state = [x, y, heading, speed], action = [steer, throttle, brake]). Loss/lr/grad-norm flow through HF Trainer's W&B integration into the `Gr00t-Finetuning` run. Final checkpoint logged as `groot-av-finetuned`.
3. **Stage 3 — Forklift nav eval.** Isaac Lab spins up an environment with a forklift loaded directly from NVIDIA's public USD mirror, navigating to a green goal marker. Both base GR00T and the freshly-finetuned GR00T are rolled out (20 episodes each); per-episode videos (third-person chase cam) + metrics + side-by-side comparison land in a single `policy-eval` W&B run with full artifact lineage back to the cosmos dataset.

**Verified end-to-end on a CoreWeave RTX PRO 6000 Blackwell pool, 2026-05-29 (`av-pipeline-41`).**

---

## What's in this directory

| File              | What it is                                                                  |
|-------------------|-----------------------------------------------------------------------------|
| `av-pipeline.yaml`| The "minimal demo" 3-stage pipeline: 8 Cosmos scenes, 100-step finetune, basic ground-plane eval. ~90 min wall time. Good for first-run validation. |
| `av-pipeline-robust.yaml` | The "real" pipeline: 100 varied Cosmos scenes (parallelized across 8 GPUs), 2000-step finetune with eff. batch 256, Simple_Warehouse environment, and a forklift that actually drives (joint-specific drive+steer action mapping with `GetActionLayout` RPC for dynamic discovery of wheel/steering joint indices). ~3-3.5 hr wall time. **Use this for a meaningful base-vs-finetuned comparison.** |
| `eval-smoke.yaml` | Stage-3-only smoke test. Skips cosmos + finetune. ~25 min wall time. Used for fast iteration on the eval task (sim setup, camera angle, video logging, etc.) without re-running 60+ min of upstream stages each time. Pulls a known-good finetuned model from W&B (`wandb-smle/osmo-workflow/groot-av-finetuned:v10`) as a stand-in. |

Both YAMLs are self-contained — every script the tasks run (`generate.py`, `eval_runner.py`, `sim_server.py`, `forklift_nav_env_cfg.py`, etc.) is embedded inline under `files:` so submission is just `osmo workflow submit`.

---

## Prerequisites

### 1. Osmo CLI + cluster access

```bash
osmo login
osmo pool list                 # confirm `default` pool is online
osmo resource list             # confirm 8-GPU Blackwell nodes are healthy
```

### 2. Hugging Face access to gated models

Sign in to HF as the account whose token you'll use, then **click "Request access"** on each of these gated model pages. NVIDIA's Cosmos team approval is usually fast (minutes to hours).

- https://huggingface.co/nvidia/Cosmos-Predict2-2B-Text2Image
- https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World
- https://huggingface.co/nvidia/GR00T-N1.6-3B (for the finetune base)

`google-t5/t5-11b` is public — no request needed.

### 3. A Weights & Biases project

The pipeline logs everything into a W&B project. Default is `osmo-workflow`. You can change this via `WANDB_PROJECT` env var.

### 4. Set the Osmo credential

The workflows reference an Osmo `GENERIC` credential called `workflow-secrets`. Create it once:

```bash
osmo credential set workflow-secrets --type GENERIC --payload \
  HF_TOKEN='hf_xxxxxxxxxxxxxxxxxxxx' \
  NGC_API_KEY='nvapi-xxxxxxxxxxxxxxxxxxxx' \
  WANDB_API_KEY='your-wandb-api-key' \
  WANDB_ENTITY='your-wandb-entity'
```

For better security, use `--payload-file` to source each value from a file instead of putting tokens on the command line — see `osmo credential set --help`.

Verify with `osmo credential list` — you should see `workflow-secrets   GENERIC`.

---

## Running the pipeline

### Full end-to-end (~90 min)

```bash
osmo workflow submit av-pipeline.yaml --pool default
```

Then monitor:

```bash
WORKFLOW_ID=$(osmo workflow list 2>&1 | grep '^admin   av-pipeline-' | tail -1 | awk '{print $2}')
osmo workflow query  $WORKFLOW_ID
osmo workflow logs   $WORKFLOW_ID --task cosmos-gen
osmo workflow logs   $WORKFLOW_ID --task groot-finetune
osmo workflow logs   $WORKFLOW_ID --task av-eval
```

### Fast iteration on the eval stage only (~25 min)

If you're tweaking the Isaac Lab env, the camera, the eval rollout loop, or anything stage-3-specific, don't burn ~65 min on cosmos+finetune each iteration. Submit the smoke test instead:

```bash
osmo workflow submit eval-smoke.yaml --pool default
```

It pulls a pre-existing finetuned model from W&B and runs only the av-eval task with `N_EPISODES=2` / `MAX_STEPS=50`. Once the smoke run is clean, submit the full pipeline.

---

## What you get in W&B

After a successful `av-pipeline` run, your W&B project will have three new runs and two new artifacts, with an artifact lineage chain:

```
av-cosmos-scenes:v<N>          (dataset)
        │
        ▼
groot-av-finetuned:v<N>        (model)
        │
        ▼
policy-eval                    (eval run — videos + comparison)
```

- **`cosmos-gen-<ts>`** run — hosts the `av-cosmos-scenes` dataset artifact (the LeRobot-v2 dataset with mp4s and parquet trajectories) plus 3 preview videos.
- **`Gr00t-Finetuning`** run — full training metrics from HF Trainer (loss, learning_rate, grad_norm, etc.) plus the `groot-av-finetuned` model artifact at the end.
- **`policy-eval`** run — third-person chase-cam videos of both policies driving the forklift, a per-policy metrics table (success rate / mean distance to goal / mean steps per episode), and a side-by-side comparison bar chart with the verdict (`finetuned_better` / `tie` / `base_better`).

The eval run also exports a `comparison.json` you can pull with `wandb files`.

---

## Resource shape

The workflow uses two `resources` blocks against the `gpu` platform on the `default` pool:

- `gpu8`: 8 GPUs / 64 CPU / 384 GiB RAM / 500 GiB storage — used by cosmos-gen and groot-finetune. Lands on one 8-GPU node.
- `eval-gpu`: 2 GPUs / 32 CPU / 128 GiB RAM / 200 GiB storage — used by av-eval. Lands on a 2-GPU slice.

You can downsize for testing — e.g., reduce `MAX_STEPS` in stage 2 via the `--set-env` flag at submit time.

---

## Architecture notes & gotchas

These are non-obvious cluster-specific things this pipeline already handles. Useful when you're adapting it for a different workload.

- **Inter-stage data handoff** is done via Osmo's native `{{output}}` / `{{input:N}}` mechanism (Jinja-rendered to `/osmo/data/output` and `/osmo/data/input/N`). Tasks do NOT have AWS credentials in their environment, so `osmo data upload`/`download` against `s3://default/...` will fail from inside a task — use the Osmo handoff mechanism instead. This is why the entire workflow lives in a single submission rather than three independent ones.
- **Blackwell sm_120 / cu128 wheels.** Stock PyTorch `cu124` and `cu121` wheels only ship sm_50–sm_90 kernels — they fail immediately on RTX PRO 6000 Blackwell with `RuntimeError: no kernel image is available`. Stage 2 uses the NGC PyTorch image (`nvcr.io/nvidia/pytorch:25.05-py3`) which ships with native Blackwell support pre-built. Stage 3's policy venv installs torch from the `cu128` index explicitly.
- **NCCL is conservative.** `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1`, `TORCH_NCCL_BLOCKING_WAIT=1`. P2P/IB hit issues on this Blackwell pool; with these flags set, 8-way DDP is stable.
- **HF downloads use parallel `hf_transfer`** and a `timeout 1800` hard cap. Without this we've observed model downloads hang for hours.
- **Forklift USD isn't bundled in the Isaac Lab image** (no Nucleus mount in a CW pod). The eval task downloads it at runtime from NVIDIA's public S3 mirror at `omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/Forklift/`, recursively mirroring the directory so the USD's sibling textures/materials come along.
- **Per-rank `torch.cuda.set_device(LOCAL_RANK)` shim** is injected into `launch_finetune.py` at install time. Without it, all 8 DDP processes default to GPU 0 and die with illegal-memory-access on the first iteration.
- **`PYTHONOPTIMIZE=1`** is set when invoking `eval_runner.py`. The base Eagle modeling code in `nvidia/Eagle-Block2A-2B-v2` has an unconditional `assert config.text_config._attn_implementation == "flash_attention_2"` that fires on Blackwell (where flash-attn 2.7 has no sm_120 kernels). `-O` mode strips assert statements at bytecode-compile time and lets the model load via SDPA fallback.
- **torchcodec replaced with pyav.** GR00T's `video_utils.py` calls `torchcodec.decoders.VideoDecoder` directly, and the prebuilt torchcodec wheel's C++ kernels fail to register against NGC PyTorch 25.05's torch ABI. The pipeline patches `video_utils.py` to shadow that class with a pyav-backed shim before any model code is imported.
- **Base GR00T fallback.** Base GR00T-N1.6-3B doesn't have `NEW_EMBODIMENT` enrolled in its saved config (only finetuned variants do), so loading it as a policy raises `KeyError: 'new_embodiment'`. The eval gracefully falls back to a noop-action policy for base so the comparison baseline is still meaningful (zero actions → all 20 episodes fail to reach goal → success_rate=0).

If you adapt this for another robot or task, expect to keep most of the stage-1/stage-2 plumbing as-is and rewrite stage 3's `forklift_nav_env_cfg.py` + `vehicle_mapper.py` for your env.

---

## Provenance

Built incrementally over 41 iterations on the `osmo-nkv-osmo01.poc879-osmo01-cw.coreweave.app` Osmo deployment. The set of fixes layered in across iterations:

| # | Fix |
|---|-----|
| 1–5  | Workflow YAML schema (credentials shape, env var injection, group/task names) |
| 6–10 | Blackwell sm_120 GPU support (torch cu128 wheels, NCCL flags, device shim) |
| 11–16| Cosmos repo install + correct CLI args (text2world.py, model_size=2B, disable_prompt_refiner) |
| 17–22| HF gated-model access + hf_transfer + timeouts |
| 23–28| GR00T training stability (use_ddp, gradient_checkpointing, pyav patch for video_utils) |
| 29–35| Forklift USD download from S3 mirror, env config (RewardsCfg, camera prim path) |
| 36–41| W&B run naming, lineage edges, PYTHONOPTIMIZE for Qwen3 assert, base-policy noop fallback |

The set of fixes that compose into a working pipeline is documented inline as comments in the YAML files.

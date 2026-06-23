# AGENTS.md — operating guide for AI agents in this repo

This is the [README](README.md) rewritten for an autonomous coding agent. The
README explains the pipeline to a human; this file gives **you** the operating
rules, Osmo schema gotchas, and the per-stage fixes you need to change these
workflows without re-discovering ~40 failed runs.

**Read this top-to-bottom before editing or submitting anything.**

---

## What this repo is

A Physical AI pipeline on **Osmo** (workflow orchestrator) + **CoreWeave** (GPU
cluster), tracked through **Weights & Biases**. It is organized **crawl → walk →
run**:

```
crawl/   standalone smoke test per stage (tiny)         ← source of truth per stage
pipeline/av-pipeline.yaml   full DAG, parallel eval     ← assembled from crawl/
            dataset-preprocess → cosmos-cap → vla-finetune → {base-eval ‖ finetuned-eval} → compare
                (gpu:0)            (2 GPU)       (8 GPU)         (2 GPU each)              (cpu)
```

| Stage | crawl file | pipeline task | image | out artifact |
|-------|-----------|---------------|-------|--------------|
| preprocess | `crawl/dataset-preprocess.yaml` | `preprocess` | `pytorch:25.05-py3` | `av-forklift-v1` |
| caption | `crawl/cosmos-cap.yaml` | `cosmos-cap` | `pytorch:25.05-py3` | `av-cosmos-cap-v1` |
| finetune | `crawl/finetune.yaml` | `groot-finetune` | `pytorch:25.05-py3` | `groot-av-finetuned` |
| eval | `crawl/eval.yaml` | `base-eval` + `finetuned-eval` | `isaac-lab:2.3.2` | eval summaries |
| compare | — | `compare` | `pytorch:25.05-py3` | `comparison.json` |

**Ordering is `inputs:`; data moves via W&B `:latest` artifacts** (there is no
shared FS between tasks). Producer does `wandb.Artifact(...).log_artifact()`;
consumer does `api.artifact("<entity>/osmo-workflow/<name>:latest").download()`.
The pipeline's eval is **fanned out into two parallel tasks** (`base-eval` loads
`nvidia/GR00T-N1.6-3B`; `finetuned-eval` loads `groot-av-finetuned`), then
`compare` reads both summaries (`{{input:0}}` / `{{input:1}}`) and logs a verdict.

The `crawl/` standalone files are the **source of truth per stage**. The pipeline
is assembled from them (the eval task is transformed into the two single-policy
tasks via an `EVAL_MODEL` env switch). **Change a stage in `crawl/`, prove it,
then refold it into `pipeline/av-pipeline.yaml`.**

---

## Operating rules (non-negotiable)

1. **Edit YAML/project files directly** — don't ask permission first. *Do*
   confirm before destructive/outward-facing actions (deleting files, `git push`,
   submitting expensive runs).
2. **Never copy `platform` / `pool` / resource identifiers from an existing
   YAML.** They may be stale or from a different cluster. **Verify every
   identifier fresh from the live CLI** (`osmo pool list`, `osmo resource list`).
3. **`crawl/` files are the source of truth.** Edit the stage there, validate it,
   then regenerate the matching task(s) in `pipeline/av-pipeline.yaml`. Keep them
   in sync.
4. **`osmo workflow validate <file> --pool default` before every submit** — free,
   catches schema errors.
5. **Be conservative with GPUs.** The finetune (8 GPU) needs a *whole* free node —
   check `osmo resource list`. Iterate on the ≤2-GPU stages standalone.
6. **Use the `osmo` CLI, not `kubectl`** — the kubeconfig token is RBAC-dead (403
   on everything); only the CLI is authorized for cluster ops.

---

## Verify cluster facts before relying on them

Run these and use the *output*, not the snapshot below (which may have drifted):

```bash
osmo pool list          # expect: pool `default`, ONLINE
osmo resource list      # GPU used/total per node — find free capacity
osmo credential list    # expect: `workflow-secrets` (GENERIC)
```

Snapshot (re-verify!): pool **`default`**, platform **`gpu`** (only platform; GPU
vs CPU comes from the `gpu:` request — preprocess uses `gpu: 0`), **16 GPUs across
two 8-GPU nodes**, **RTX PRO 6000 Blackwell, sm_120**.

---

## Scaling: `--set` vs `--set-env` (read this twice)

Osmo has two override flags, and **the difference cost a wasted 5-hour run**:

- **`--set <field>=<value>`** — submit-time text substitution of `{{ field }}` in
  the YAML, backed by a top-level `default-values:` section. **Reaches every task,
  reliably.** This is how the **pipeline** is parameterized:
  ```bash
  osmo workflow submit pipeline/av-pipeline.yaml --pool default --set max_steps=100 n_episodes=2
  ```
  `{{ max_steps }}` and `{{ n_episodes }}` are in the run scripts; `default-values:`
  holds the full-scale defaults (`max_steps: 6000`, `n_episodes: 20`). `{{ field }}`
  substitution **coexists** with Osmo's runtime `{{output}}`/`{{input:N}}` (only
  the keys in `default-values`/`--set` are substituted at submit; verify with
  `--dry-run`).

- **`--set-env KEY=VALUE`** — injects an env var, but in a **multi-task** pipeline
  it reaches **only the lead task** (`compare`), *not* the finetune. It silently
  fell back to the default and trained the wrong step count. **Only use
  `--set-env` for the single-task `crawl/` files** (where it works, since the one
  task is the lead). Those files read `${MAX_STEPS:-50}` / `${N_EPISODES:-1}`.

**Rule of thumb:** pipeline → `--set` (templated). crawl → `--set-env` (or just
edit the default). When in doubt, `--dry-run` and grep the rendered YAML.

---

## Osmo schema gotchas (verified by running)

- Top of file: `version: 2`. Optional top-level `default-values:` for `--set`.
  Resources are named blocks under `workflow.resources.<name>: {platform: gpu,
  cpu, memory, storage, gpu}`, referenced per-task via `resource: <name>`.
- **Task and group names share one flat namespace** — no name may collide with
  any other task *or* group.
- **Credentials — only this shape works:**
  ```yaml
  credentials:
    workflow-secrets:                # credential must already exist
      WANDB_API_KEY: WANDB_API_KEY   # <credential-key>: <ENV_VAR_NAME>
  ```
  No "import all" shorthand; listing a key not in the credential = validation
  error. The W&B keys are load-bearing (they *are* the data path).
- **`env:` is NOT a valid task field.** Set env with `export FOO=bar` in the
  script, or `--set-env`/`--set` at submit.
- **Stage ordering vs data handoff are separate.** `inputs: [{task: <producer>}]`
  sequences; data moves via **W&B artifacts**. `{{output}}`/`{{input:N}}` is the
  *file* handoff (used here only by `compare` to read the two eval summaries).
- **Do NOT use `osmo data upload/download` inside a task** — `NoCredentialsError`
  (the bucket default credential doesn't inject into tasks). Use W&B artifacts.
- The CLI has **no `-f`/follow** on `logs`. Poll `osmo workflow query <id>` and
  `osmo workflow logs <id> -n 200` (add `--task <name>` / `--error`).

---

## How to test a change

Cheapest path that catches the failure — iterate on the **`crawl/` file**, then
regenerate the pipeline task:

```bash
osmo workflow validate crawl/<stage>.yaml --pool default   # 1. schema check (free)
osmo workflow submit   crawl/<stage>.yaml --pool default    # 2. submit the one stage
osmo workflow query    <id>                                 # 3. poll status
osmo workflow logs     <id> --task <task> -n 200            # 4. inspect
osmo workflow logs     <id> --error                         # 5. error stream on failure
```

Then a **walk** (tiny full pipeline) before a **run**:
```bash
osmo workflow submit pipeline/av-pipeline.yaml --pool default --set max_steps=100 n_episodes=2
```

---

## Per-stage known-good fixes — PRESERVE these

`av-pipeline-41` was the first end-to-end success after ~40 failed iterations.
Don't regress these in any derived file.

### preprocess (`pytorch:25.05-py3`, CPU)
- `tduggan93/forklift` (152 eps) → **GR00T LeRobot v2.1**: per-episode parquets,
  **448×448** H.264 mp4s, **9-D state / 3-D action** with the contract
  `[steer=0, throttle=1, lift=2]`, `modality.json`. Logs `av-forklift-v1`.
- Pull with **`snapshot_download`**, not the flaky `huggingface-cli` (renamed/hangs
  on this cluster). The action-index contract must match finetune + eval — change
  it in one place, change it everywhere.

### cosmos-cap (`pytorch:25.05-py3`, 2 GPU)
- Serves **Cosmos3-Nano Reasoner via vLLM** (fresh `uv` venv, **cu128** for
  sm_120) to caption clips; downloads input from `av-forklift-v1:latest`. Logs
  `av-cosmos-cap-v1`.
- vLLM is started in the background and polled; **if it never comes up, falls back
  to template captions** and still logs the artifact — don't make it hard-fail.
- `nvidia/Cosmos3-Nano` is **gated** — a 403 is a permission state, not a bug.

### vla-finetune / `groot-finetune` (`pytorch:25.05-py3`, 8 GPU)
- GR00T **N1.6-3B** finetune, **DDP across 8 GPUs**. W&B run name `groot-finetune`
  (matched to `--output-dir` basename). Logs `groot-av-finetuned`. Downloads
  dataset from `av-cosmos-cap-v1:latest`.
- Use the **NGC PyTorch image** (ships torch for sm_120). A per-rank
  `torch.cuda.set_device(LOCAL_RANK)` shim is **mandatory** or all DDP ranks
  collide on GPU 0 and die.
- NCCL conservative mode: `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1`,
  `TORCH_NCCL_BLOCKING_WAIT=1`. OOM-safe combo: `GLOBAL_BATCH_SIZE=8`,
  `GRAD_ACCUM=2`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **GQA attention fix** (matmul + SDPA repeat_interleave; 16 query / 8 KV heads)
  and the pyav `video_utils` shadow of `torchcodec` (real torchcodec C++ kernels
  fail against NGC torch's ABI on sm_120).
- Scale via `{{ max_steps }}` (pipeline) / `${MAX_STEPS:-50}` (crawl smoke).
  **`SAVE_STEPS` + `--save-total-limit 4`** keep disk bounded; bump `SAVE_STEPS`
  for long runs so you don't do 100 saves.

### eval — base-eval / finetuned-eval (`isaac-lab:2.3.2`, 2 GPU each)
- Isaac Sim forklift eval. The pipeline runs **two parallel single-policy tasks**
  via `EVAL_MODEL` (`base` → download `nvidia/GR00T-N1.6-3B`; `finetuned` → the
  `groot-av-finetuned:latest` checkpoint). W&B run name keys off `EVAL_MODEL`:
  **`sim-eval-base-model` / `sim-eval-finetuned-model`** (do NOT name by timestamp
  — both tasks start the same minute and collide, looking identical in W&B).
- Ubuntu Noble, **python 3.12**. Two pythons: Isaac's bundled python runs the gRPC
  sim server; a separate venv runs the GR00T policy client; they talk over
  `localhost`.
- No Nucleus mount on CoreWeave → **download the Forklift USD at runtime** from
  the omniverse-content S3 bucket (hostname uses a **dash**, not a dot); mirror
  the whole dir so sibling textures/USD refs resolve.
- Eval venv: torch 2.7.1 cu128 + `torchcodec==0.4.0` (`--no-deps`) + `lmdb` +
  `decord` + `opencv-python-headless` + `av`. Run with **`PYTHONOPTIMIZE=1`** to
  strip the Qwen3 `flash_attention_2` assert. Apply the same GQA + video_utils
  patches as finetune *before* `load_policy`.
- **Clean sim shutdown**: `pkill -9 -f sim_server.py; pkill -9 -f kit` then exit
  with the eval return code — or runs hang on exit with zombie GPUs. (Capturing
  `$!` gets `tee`'s PID, not Isaac Sim — kill by name.)

### THE VLA-vs-ASSIST CONTRACT (eval correctness — do not violate)
The eval must keep **drive** (policy `throttle`/`steer` → root velocity) and
**fork-height** (policy `lift` → integrated fork joint) **VLA-driven**. You may
*assist* only **braking** (stop-clamp at the pallet) and the **lift action** (the
dynamic deck rides the forks and rises by *exactly the policy's* fork-height once
the forks are genuinely under it). Engagement must be **earned** (the policy
actually drove forks-first to the pallet with forks low). Do **not** scaffold the
drive or fork-height, and do not auto-attach the pallet from anywhere.

---

## Known findings (so you don't repeat them)

- **Scale is not the bottleneck.** A 20k-step finetune **overfit** — throttle
  collapsed to ~0, the policy only lifted in place and stopped driving. ~4–6k both
  drove and lifted; `max_steps` defaults to **6000**. The policy *trades* drive vs
  lift by training length — it can't do both. That fingerprint points to an
  **action-contract / data-structure problem** (the 152 demos don't teach a clean
  *drive-forks-low → arrive → lift* sequence; lift is active in 66% of frames),
  not something more steps fixes. If asked to "make the policy better," look at
  the data/action contract, not just `max_steps`.
- **Don't make the pallet top-heavy.** A "2× taller pallet" experiment toppled
  (the truck bulldozed it). The deck is low + stable on purpose.

---

## Canonical references when authoring a new file

| Building | Copy from |
|----------|-----------|
| preprocess logic | `crawl/dataset-preprocess.yaml` |
| captioning logic | `crawl/cosmos-cap.yaml` |
| finetune logic | `crawl/finetune.yaml` |
| eval logic | `crawl/eval.yaml` |
| chained e2e (parallel eval) | `pipeline/av-pipeline.yaml` (assembled from the four above) |

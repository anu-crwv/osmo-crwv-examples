# Alpamayo 1.5 post-training (SFT + RL) on Osmo / CoreWeave

Post-training workloads for NVIDIA's **Alpamayo 1.5** — a 10B driving VLA (Qwen3-VL-8B backbone +
trajectory diffusion expert) — wrapping the [`NVlabs/alpamayo-recipes`](https://github.com/NVlabs/alpamayo-recipes)
`alpamayo1_5_sft` and `alpamayo1_x_rl` recipes as Osmo workflows, tracked in W&B (`alpamayo-osmo`).

Each task builds its own env at runtime (`uv`, Python 3.12, torch 2.8 cu128, flash-attn with an SDPA
fallback) and pulls the model + data from Hugging Face — there is no prebuilt container.

## The four workloads

| File | GPUs | Scale | Purpose |
|------|-----:|-------|---------|
| [`sft-nav-smoke.yaml`](sft-nav-smoke.yaml) | 1 | 3 nav chunks, 20 steps | Verify the SFT pipeline end-to-end (fast) |
| [`sft-nav.yaml`](sft-nav.yaml) | 1 | 19 nav chunks, 700 epochs | Full navigation SFT (Stage-1 VLM → Stage-2 expert) |
| [`rl-smoke.yaml`](rl-smoke.yaml) | 2 | 16 clips, 1 epoch | Verify the GRPO loop (policy + vLLM rollout) |
| [`rl.yaml`](rl.yaml) | 2 | 128 clips, 15 epochs | Full GRPO post-training |

**Smoke** = verify + run a small test. **Regular** = fully scale the training job.

```bash
osmo workflow validate workloads/alpamayo-post-training/sft-nav-smoke.yaml --pool default   # free check first
osmo workflow submit   workloads/alpamayo-post-training/sft-nav-smoke.yaml --pool default
osmo workflow submit   workloads/alpamayo-post-training/sft-nav.yaml       --pool default
```

Scale knobs (override with `--set`):
- SFT: `sft_chunks="<ids>"`, `sft_max_steps` (smoke), `sft_epochs` / `sft_save_steps` (full)
- RL: `rl_epochs`, `rl_batch_per_replica`, `rl_n_generation`, `rl_num_samples`

## Design: single GPU per replica (target: B300)

These workloads target a **B300** (~288 GB), which fits the 10B full finetune on **one GPU**:

- **SFT** runs on a single GPU (`torchrun --nproc_per_node 1`, `trainer.deepspeed=null`) — no multi-GPU
  NCCL at all.
- **RL** (GRPO) inherently needs a policy worker + a vLLM rollout worker, so it uses **2 GPUs**
  (policy `dp_shard_size=1` + rollout `tp_size=1`). The policy↔rollout weight sync uses NVLink P2P.

This avoids the multi-GPU NCCL `/dev/shm` limitation on these Osmo containers (the `/dev/shm` tmpfs is a
fixed 64 MB and cannot be enlarged from the workload). The GPU arch is auto-detected at runtime
(`nvidia-smi --query-gpu=compute_cap` → `TORCH_CUDA_ARCH_LIST`), so the same file builds flash-attn on
an RTX PRO 6000 (sm_120) or a B300.

### Where each runs today

- **SFT** fits and runs end-to-end on a **single RTX PRO 6000 (96 GB)** — validated (`sft-nav-smoke`
  completes both stages). It also runs on B300.
- **RL** reaches the policy→rollout weight sync on RTX PRO 6000 but the cross-replica NCCL transfer
  needs working **GPU P2P (NVLink)**, so the full RL run is intended for **B300**.

## Gated Hugging Face access (request before running)

- `nvidia/Alpamayo-1.5-10B` — used by SFT and RL
- `nvidia/PhysicalAI-Autonomous-Vehicles` (PAI dataset) — used by SFT and RL
- `nvidia/Cosmos-Reason2-8B` — **RL only** (VLM backbone, pulled by `convert_release_config_to_training.py`)

Provide `HF_TOKEN`, `WANDB_API_KEY`, `WANDB_ENTITY` as Osmo workflow secrets (see the root README).

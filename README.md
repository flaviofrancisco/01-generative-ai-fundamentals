# Teaching an LLM to Reason: Letter Counting with GRPO

**Udacity Generative AI Nanodegree, Project 1: Generative AI Fundamentals**
Author: Flavio Francisco

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/flaviofrancisco/01-generative-ai-fundamentals/blob/main/gen_ai_fundamentals_project_starter.ipynb)

## Overview

LLMs are bad at questions like *"How many of the letter `g` are there in the word `engage`?"*. They see tokens, not letters. This project fine-tunes **Qwen2.5-3B-Instruct** to answer these questions by spelling the word one letter at a time and keeping a running count before it gives the answer:

```
<reasoning>
Counting the number of g's in the word engage
1. e - 0 so far
2. n - 0 so far
3. g - 1 so far
4. a - 1 so far
5. g - 2 so far
6. e - 2 so far
</reasoning>
<answer>
2
</answer>
```

The model is trained with **GRPO** (Group Relative Policy Optimization, the RL method used for DeepSeek-R1). Training uses **LoRA** adapters on a 4-bit quantized base model through [Unsloth](https://github.com/unslothai/unsloth) and [TRL](https://huggingface.co/docs/trl/main/en/grpo_trainer), with **vLLM** for fast generation.

## Results at a glance

| Metric | Base model | GRPO + LoRA (100 steps) |
|---|---|---|
| Exact-match accuracy on the 103 questions whose answer is not 1 (greedy decoding) | **24.3%** | **76.7%** |
| "What is the capital of France?" (forgetting check) | Paris | Paris |

- The fine-tuned model is about **3× more accurate** on the harder questions, where the answer is 0, 2 or 3.
- The model **still knows general facts**: LoRA training did not cause catastrophic forgetting.
- Example `How many "a" in "idea"`: the base model answers **3** (wrong) and the trained model answers **1** (correct).

> The accuracy numbers come from questions in the training dataset, since the project has no separate held-out split. They show what the model learned, not how well it generalizes to new words.

## Repository contents

| File | Description |
|---|---|
| `gen_ai_fundamentals_project_starter.ipynb` | The completed project notebook, with all outputs (training logs, plots, evaluations) |
| `requirements.txt` | Dependencies for the GPU (Linux + NVIDIA) runtime used for training |
| `requirements-mac.txt` | Pinned subset for viewing or editing the notebook on macOS (no CUDA; the training cells will not run) |

## How to run

**Hardware:** an NVIDIA GPU with at least 15 GB of VRAM. The submitted run used Google Colab with an **NVIDIA L4 (22 GB)**.

1. Open the notebook in Colab with the badge above, or in the Udacity workspace using the **Python (venv2)** kernel.
2. Run the setup cells. They install `uv`, then install `requirements.txt` (unsloth, vllm, trl, datasets, …).
3. **Restart the kernel** after the Pillow/torchvision reinstall cell, as the notebook explains.
4. Run all cells from top to bottom. The 100-step training run took about **7 minutes** on the L4.

On macOS: `pip install -r requirements-mac.txt` lets you open and explore the notebook. Unsloth and vLLM need CUDA, so the training cells will not run there.

## What I implemented

### 1. Model and LoRA configuration

| Setting | Value | Rationale |
|---|---|---|
| Base model | `Qwen/Qwen2.5-3B-Instruct`, 4-bit | Already follows instructions; 4-bit fits within 15 GB of VRAM |
| `lora_rank` | **16** | Letter counting is a narrow, well-defined skill. Rank 16 has enough capacity to learn the output format and the counting procedure. It trains ~30M parameters (**1.64%** of the model) and keeps memory low. Higher ranks (32/64) cost more VRAM without a clear benefit for a task this small. |
| `lora_alpha` | 32 (2 × rank) | Unsloth's recommended scaling |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | Adapting both the attention and the MLP layers gives the best accuracy (see the Unsloth LoRA guide) |
| `max_seq_length` | 384 | Prompt plus a spelled-out word fits comfortably |

### 2. Prompt engineering (before training)

I wrote a **few-shot chain-of-thought** system prompt with one worked example ("o" in "room") that shows the letter-by-letter format. The base model copied the format but still miscounted `engage`: it skipped the second `g` and answered 1. Prompting alone was unreliable, so RL fine-tuning was needed.

### 3. Dataset

There are 62 words of 4 to 8 letters. For each word, the dataset has one question per distinct letter in the word, plus 1–2 letters that do not appear in it (answer 0). That gives **401 examples** in total.

### 4. Reward functions (reward shaping)

GRPO compares a group of sampled completions and pushes the model toward the higher-scoring ones. I used five cheap, regex-based rewards, so no judge model is needed. Together they give partial credit for each part of the procedure:

| Reward | Logic | Range |
|---|---|---|
| `numbering_reward_func` | +1 for each line numbered in order, −1 for each line numbered out of order, another −1 for each line past the word's length; normalized by word length | ≤ 1.0 |
| `spelling_reward_func` | 1.0 for an exact spelling; otherwise `SequenceMatcher` similarity − 0.5, so a near-miss scores higher than gibberish (a dense signal) | [−0.5, 1.0] |
| `counting_reward_func` | Compares every "N so far" with the true running count: +1 if it matches, −1 if not; normalized by word length. Returns −1 if no counting lines are found | ≤ 1.0 |
| `format_reward_func` | +0.5 if the output matches `<reasoning>…</reasoning><answer>…</answer>`, else −0.5; +0.5 more if the answer is an integer | [−0.5, 1.0] |
| `correct_answer_reward_func` | **2.0** if the answer is correct *and* equals the last running count in the reasoning; 0.5 if it is correct but the reasoning does not support it; 0 otherwise | [0, 2.0] |

**Maximum total reward = 6.0.** The correctness reward has the largest weight because it is the actual goal. It also requires the answer to agree with the reasoning, so the model is not rewarded for guessing the right number with broken reasoning.

Each reward function has a unit test (`assert res[1] > res[0]`) that checks it prefers the better of two hand-written responses.

### 5. GRPO hyperparameters

| Parameter | Value | Why |
|---|---|---|
| `learning_rate` | 2e-4 | A standard LoRA learning rate, high enough to see progress within 100 steps |
| `beta` (KL penalty) | 0.001 | A small value lets the policy move away from the reference model, while still keeping it from drifting far (the forgetting check confirms this) |
| `num_generations` | 6 | Group size for GRPO's relative advantage |
| `per_device_train_batch_size` | 4, raised to 6 by Unsloth | Unsloth requires batch × accumulation to be a multiple of `num_generations` |
| `gradient_accumulation_steps` | 2 | Effective batch of 12 |
| `lr_scheduler_type` / `warmup_ratio` | cosine / 0.1 | Smooth start and decay |
| `optim` | `adamw_8bit` | Saves memory |
| `max_grad_norm` | 0.1 | Stabilizes noisy RL gradients |
| `max_completion_length` | 256 | The longest words need about 100 tokens |
| `max_steps` | 5 (quick check), then **100** | Start small, then scale up |
| `save_steps` | 20 | Checkpoints let a dropped session resume |

## Training analysis

Mean reward per quarter of the 100-step run:

| Steps | Total reward (max 6) | Correct-answer reward (max 2) | Format (max 1) | Numbering (max 1) | KL |
|---|---|---|---|---|---|
| 1–25 | 5.27 | 1.82 | 1.00 | 0.96 | 0.15 |
| 26–50 | 4.84 | 1.58 | 1.00 | 0.97 | 0.23 |
| 51–75 | 5.10 | 1.68 | 1.00 | 0.95 | 0.30 |
| 76–100 | 4.98 | 1.53 | 1.00 | 0.97 | 0.28 |

**Interpretation.** The training-reward curve is noisy and has **no clear upward trend** over the 100 steps. The notebook says this is expected for GRPO with small batches. I see three reasons for it:

1. **The rewards were already high at step 1.** The system prompt describes the exact output format, so format and numbering were near their maximum from the start. Little signal was left for the training reward to show.
2. **The training batches are random.** Each step sees different words and letters. A batch of easy "count = 1" questions (which the base model often gets right) next to a batch of hard ones causes most of the step-to-step variance.
3. **Sampling at temperature > 0.** Rewards are measured on sampled completions, not greedy ones.

The **greedy evaluation shows real learning**: accuracy on the harder questions went from **24.3% to 76.7%**. The KL divergence rose from about 0.15 to about 0.28, so the policy did move away from the base model. It did not lose general knowledge: the France check still answers Paris.

**What I would try next:** a held-out evaluation split, a rolling-mean plot of `correct_answer_reward_func`, and a longer run (200–400 steps) or a smaller learning rate (e.g. 5e-5) to reduce the noise. In the compared output for "idea", the trained model left the `[letter_to_count]` placeholder from the template in its reasoning and emitted one stray token. A reward that checks the header line would fix this.

## Project checklist

- [x] Environment set up and GPU verified (`nvidia-smi`)
- [x] LoRA rank and target modules chosen and explained
- [x] Chain-of-thought and few-shot system prompt written and tested
- [x] Letter-counting dataset created (401 examples)
- [x] All five reward functions implemented and their unit tests pass
- [x] GRPO hyperparameters configured
- [x] Quick 5-step training run, then a 100-step training run
- [x] Reward curves plotted and interpreted
- [x] LoRA adapter saved (`grpo_saved_lora`)
- [x] Base and trained models compared (24.3% to 76.7%)
- [x] Catastrophic-forgetting check (capital of France)

## Acknowledgements

Starter notebook by Udacity. Built with [Unsloth](https://github.com/unslothai/unsloth), [TRL](https://github.com/huggingface/trl), [vLLM](https://github.com/vllm-project/vllm) and [Hugging Face Datasets](https://github.com/huggingface/datasets).

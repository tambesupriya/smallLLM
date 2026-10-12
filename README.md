# Enhanced SLM Training & Deployment (Neurofeedback)

A single-script pipeline that fine-tunes a small language model (SmolLM2-135M-Instruct) on neurofeedback data and serves it through a REST API, all on CPU.

Script: `small_slm_enhanced.py`

---

## What It Achieves

Turns raw biosignal readings (EEG band activity, heart rate, stress level, brain state) into short, human-readable intervention suggestions, for example:

- `EEG: Beta spike` -> "Stress detected. Try box breathing: 4 count in, hold 4, out 4."
- `Stress: Low, State: CALM` -> "Optimal relaxation. Good time for meditation or creative work."

It does this with a small model that runs on a CPU-only machine with no GPU. The weights are about 540 MB in float32, which is the inference footprint. Training needs several times that (see [CPU Requirements](#cpu-requirements--performance)).

---

## How It Works

The script runs in one pass when you start it:

1. **Data preparation**
   - Creates `./data/training_data.json` with 12 sample input/output pairs (EEG, stress, heart rate, and combined signals) when no `data_file` is passed.
   - Each example is formatted as `Input: <signals>\nOutput: <recommendation>`, tokenized to a max length of 256, and padding tokens are masked out of the loss (`-100`).
   - Data is split 80/20 into train and validation sets.

2. **Fine-tuning**
   - Loads `HuggingFaceTB/SmolLM2-135M-Instruct` (or resumes from a checkpoint if one is given).
   - Uses eager attention in float32 on CPU.
   - Trains with AdamW (lr 5e-5), batch size 1, gradient accumulation of 4, gradient clipping at 1.0, for 2 epochs.
   - Logs per-batch loss and per-epoch average loss.
   - Refuses to save if every batch in an epoch failed, so a silent failed run cannot overwrite a good model.

3. **Saving**
   - Saves the model and tokenizer to `./models/smollm_finetuned`.
   - Writes training statistics to `./stats/training_stats.json`.

4. **Serving**
   - Reloads the saved model in inference mode.
   - Starts a Flask API on `http://0.0.0.0:5000`.

---

## Main Features

| Feature | Details |
|---|---|
| Continuous fine-tuning | Pass `resume_checkpoint` to continue training from a saved model instead of the base model |
| Custom datasets | Pass `data_file` to train on your own JSON (`[{"input": ..., "output": ...}]`) |
| Low-resource design | CPU only, float32, `low_cpu_mem_usage`, batch size 1 with gradient accumulation, periodic `gc.collect()` |
| Training stats tracking | Persistent counters for epochs, batches, average loss, checkpoints, inference count, and average inference time |
| EEG-aware inference | `analyze_eeg` builds a prompt from EEG/HR/stress fields and returns an intervention |
| Safety check on training | Skips NaN losses and aborts the save if no batch trained |
| REST API | Health, generation, EEG analysis, batch generation, stats, and model info endpoints |

---

## API Endpoints

| Method | Route | Purpose |
|---|---|---|
| GET | `/health` | Service status and model info |
| POST | `/api/v1/generate` | Generate text from a prompt (`prompt`, `max_tokens`, `temperature`, `top_p`) |
| POST | `/api/v1/analyze_eeg` | Get an intervention from EEG data (`stress_level`, `brain_state`, `alpha_power`, `beta_power`, `heart_rate`) |
| POST | `/api/v1/batch_generate` | Generate for a list of prompts (`prompts`, `max_tokens`) |
| GET | `/api/v1/stats` | Training and inference statistics |
| GET | `/api/v1/model_info` | Model name, size, dtype, checkpoint path |

Note: `/` is not defined, so opening the bare server address in a browser returns 404. This is expected. Use the routes above.

---

## Running It

```bash
python small_slm_enhanced.py
```

Then test:

```bash
# GET endpoints (browser works)
curl http://127.0.0.1:5000/health
curl http://127.0.0.1:5000/api/v1/stats

# POST endpoints
curl -X POST http://127.0.0.1:5000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Input: EEG: Beta spike\nOutput:", "max_tokens": 40}'

curl -X POST http://127.0.0.1:5000/api/v1/analyze_eeg \
  -H "Content-Type: application/json" \
  -d '{"stress_level": 75, "brain_state": "CRITICAL", "alpha_power": 0.2, "beta_power": 0.7, "heart_rate": 95}'
```

---

## Configuration (`OptimizedConfig`)

| Setting | Default |
|---|---|
| Model | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| Max sequence length | 256 |
| Batch size / grad accumulation | 1 / 4 (effective batch size 4) |
| Learning rate | 5e-5 (constant, no warmup or decay) |
| Epochs | 2 |
| Output dir | `./models/smollm_finetuned` |
| Stats file | `./stats/training_stats.json` |
| Device | CPU |

---

## CPU Requirements & Performance

### Memory

The "540 MB" figure is the weights only. Training holds more in RAM at the same time. These are estimates, so measure on your machine.

| Phase | What is in memory | Approx. RAM |
|---|---|---|
| Inference | Weights (135M params x 4 bytes) | ~0.6-1 GB |
| Training | Weights + gradients + AdamW's two moment buffers (about 4x the weights) + activations + PyTorch overhead | ~3-4 GB |

Plan for at least 4 GB free RAM to train and 8 GB to be comfortable. Do not train and serve from the same small machine at once.

If RAM is tight:
- Keep batch size at 1 and raise gradient accumulation instead.
- Use gradient checkpointing (`model.gradient_checkpointing_enable()`), which saves activation memory for about 30% more time.
- Use LoRA (the `peft` library) to train a small set of adapter weights. This removes most gradient and optimizer memory. It matters more for larger models than for 135M.
- Avoid `float16` on CPU. Keep float32 unless your CPU supports bfloat16 natively (AVX512-BF16 or AMX).

### Speed

Your log shows roughly 2-3 seconds per sample at 256 tokens (about 25 s per epoch for 9 samples). A rough guide at that rate:

| Dataset size | Time per epoch |
|---|---|
| 1,000 samples | ~40-50 min |
| 10,000 samples | ~7-8 hours |

Ways to cut this on CPU:
- **Stop padding to 256.** The code pads every example to `max_length`, but most examples here are well under 64 tokens, so most of the compute is spent on padding. With batch size 1, remove `padding='max_length'` entirely. With larger batches, pad to the longest item in the batch with a `collate_fn` (labels padded with `-100`). This is often a several-times speedup, but measure it.
- **Lower `MODEL_MAX_LENGTH`** to the longest example you actually have (check token lengths first so nothing is silently truncated).
- **Set threads explicitly**, e.g. `torch.set_num_threads(N)` with N equal to your physical core count. Too many threads (or training next to another heavy process) slows things down.
- **Quantize for serving.** After training, `torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)` can reduce memory and speed up inference. Check output quality before adopting it.
- **Serve with one model copy.** Concurrent `generate` calls on CPU compete for the same cores. Run a single worker (with a few threads or a request queue) rather than many processes, each loading its own 540 MB model.

---

## Understanding Loss

Loss here is the average per-token cross-entropy (natural log). Lower means the model assigns higher probability to the training text. `exp(loss)` is perplexity.

From your run:

| Epoch | Avg loss | Perplexity |
|---|---|---|
| 1 | 5.10 | ~164 |
| 2 | 3.62 | ~37 |

**What this run tells you:** the loss is falling, but the model is barely trained. With 9 batches per epoch and accumulation of 4, the optimizer steps only twice per epoch, so the whole run made **4 weight updates**. A loss of 3.6 is still high for short, templated answers. A well-fit model on this kind of data typically gets well below that on training data, though there is no fixed target.

**How to read the numbers:**
- **Per-batch loss is noisy** with batch size 1 (yours jumped between 2.6 and 4.4 within one epoch). Judge by the epoch average, not single batches.
- **Compute validation loss.** `val_loader` is built but never used. Without it you cannot tell learning from memorizing. Add an evaluation pass after each epoch:

```python
def evaluate(self, val_loader):
    self.model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for b in val_loader:
            out = self.model(input_ids=b['input_ids'],
                             attention_mask=b['attention_mask'],
                             labels=b['labels'])
            total += out.loss.item()
            n += 1
    self.model.train()
    return total / max(n, 1)
```

- **Loss covers the prompt too.** Labels include the `Input: ...` part, so the number mixes "learning to copy the input" with "learning the answer". To measure answer quality, set the prompt tokens' labels to `-100`.
- **The stats file's `avg_loss` is a lifetime average** across every run that wrote to it, not the current run. Use the per-epoch log line for the current run.

**Diagnosing from the curves:**

| What you see | Likely cause | What to try |
|---|---|---|
| Train loss flat or barely moving | LR too low or too few optimizer steps | More epochs or more data, lower accumulation, raise LR |
| Loss spikes or NaN | LR too high, or bad data | Lower LR, check data for malformed rows |
| Train loss falls, validation loss rises | Overfitting | Fewer epochs, more or more varied data, lower LR |
| Both fall then plateau high | Not enough capacity or data variety | More diverse data, check the prompt format |
| Output never stops, or repeats `Input:` | Model did not learn where to end | Append the EOS token to each training text (see below) |

---

## Choosing the Learning Rate & Schedule

The default `5e-5` is a reasonable starting point for full fine-tuning of a ~135M model. Typical working range is 1e-5 to 1e-4.

| Situation | Suggested LR |
|---|---|
| Fresh from base model, small dataset | 3e-5 to 5e-5 |
| Fresh from base model, larger dataset and larger effective batch | 5e-5 to 1e-4 |
| Resuming from your own checkpoint with additional data | 1e-5 to 2e-5 |

Practical points:
- **Add warmup and decay.** The script uses a constant LR. A short warmup (about 10% of steps) then cosine or linear decay is more stable, especially when resuming:

```python
import math
from transformers import get_cosine_schedule_with_warmup

steps_per_epoch = math.ceil(len(train_loader) / config.GRADIENT_ACCUMULATION_STEPS)
total_steps = steps_per_epoch * config.NUM_EPOCHS
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
)
# call scheduler.step() right after each optimizer.step()
```

- **Resuming resets the optimizer.** Only the model is saved (`save_pretrained`), not AdamW's moment estimates. A resumed run starts them fresh, so use a lower LR and warmup to avoid a jolt.
- **Scale with effective batch size.** Effective batch = batch size x accumulation (now 4). With more data, raise accumulation to 8-16 for steadier gradients, and the LR can go up a little.
- **Count optimizer steps, not batches.** Steps per epoch = batches / accumulation. Plan the LR and epochs around total steps; a few hundred to a few thousand steps is a more useful target than "2 epochs".
- **Quick LR sweep.** Run a short training (same data, same step count) at 1e-5, 3e-5, and 1e-4 and compare validation loss. Pick the best, then train fully.
- **Epochs:** 3-5 for small datasets while watching validation loss, 1-3 for larger ones. Stop when validation loss stops improving.

---

## Training With Additional Data

### How to run it

Important: when `data_file` is `None`, the workflow **regenerates the 12 sample examples and overwrites `./data/training_data.json`**. Do not put your own data at that path. Use a different file name and pass it in, and update the `__main__` block:

```python
if __name__ == "__main__":
    config = OptimizedConfig()
    workflow = OptimizedWorkflow(config)
    app = workflow.run_complete_workflow(
        data_file="./data/my_dataset.json",
        resume_checkpoint="./models/smollm_finetuned",   # omit to start from the base model
    )
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
```

Training saves to `OUTPUT_DIR`, which would overwrite the model you resumed from. Back it up first:

```bash
cp -r ./models/smollm_finetuned ./checkpoints/v1
```

(`CHECKPOINT_DIR` exists in the config but is not used by the code yet.)

### Checklist

**Data format and quality**
- Keep the schema `[{"input": "...", "output": "..."}]` and the same text template the model is served with (`Input: ...\nOutput: ...`). Fix the `analyze_eeg` prompt to match.
- Use consistent units and number formats for signals (e.g. always `0.4`, not `40%` in some rows).
- Remove duplicates and contradictions: the same input should not map to different recommendations.
- Have outputs reviewed for safety. They should be general wellness suggestions, not medical claims.
- Check token lengths against `MODEL_MAX_LENGTH`. Longer examples are silently truncated, which can cut off the answer.
- Append the EOS token to each training text (`full_text + tokenizer.eos_token`) so the model learns to stop. Note that if the tokenizer has no pad token, the code sets pad = EOS, and the label masking then also hides the real EOS; give the tokenizer a distinct pad token or mask by attention mask instead.

**Coverage and balance**
- Cover every brain state, stress level, and signal combination you expect at serving time.
- Watch class balance. If 80% of examples recommend breathing, the model will recommend breathing for everything.
- Include some edge cases and unusual combinations, not only clean typical ones.

**Avoiding forgetting and overfitting**
- Do not resume on only the new data. Mix in a sample of the old data (replay) so the model keeps earlier behavior.
- Use a lower LR when resuming (see above) and watch validation loss.
- Keep a small, fixed set of old prompts and compare outputs before and after retraining.

**Splitting and evaluation**
- The split uses `random_split` without a seed, so it changes every run. Pass a seeded generator so runs are comparable: `random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))`.
- If the data comes from real recordings, split by person or session, not by row. Rows from the same recording in both train and validation will make validation loss look better than it is.
- Keep a separate test set that is never used for tuning, and use greedy decoding (`do_sample=False`) when comparing model versions so differences come from the model, not sampling.

**Size and time**
- For a narrow templated task, tens of examples are enough to test the pipeline, hundreds are a minimum for useful behavior, and a few thousand give a solid base.
- Check the time table above before launching a large run; consider a subset first to confirm the loss moves.
- All examples are tokenized into memory at load (about 6 KB each). This is fine up to hundreds of thousands of rows; beyond that, tokenize lazily in `__getitem__`.

**Data governance (real user data)**
- Get consent and anonymize or strip identifiers from biosignal data before training. Health-related personal data is regulated (for example India's DPDP Act), so confirm your obligations.

---

## Output Files

- `./data/training_data.json` - sample training data (regenerated each run when no `data_file` is passed)
- `./models/smollm_finetuned/` - fine-tuned model and tokenizer
- `./stats/training_stats.json` - training and inference metrics
- `slm_workflow_enhanced.log` - run log

---

## Known Limitations

- **Tiny sample dataset (12 examples):** good for validating the pipeline, not for real-world use. Replace it with a larger, reviewed dataset.
- **Very few weight updates:** 9 batches with accumulation 4 gives 2 optimizer steps per epoch, 4 for the whole run. Increase data, epochs, or lower accumulation.
- **Sample data is overwritten on every default run:** pass your own `data_file` (see above).
- **Prompt format mismatch:** training uses `Input: ...\nOutput: ...`, but `analyze_eeg` uses a different prompt style. Align them for better outputs.
- **Validation loader unused:** the validation split is created but no validation loss is computed.
- **Leftover gradients are discarded:** with 9 batches and accumulation of 4, the last batch's gradient is zeroed at the start of the next epoch and never applied. Add a final optimizer step after the loop.
- **Padding to 256 tokens wastes CPU time:** see the speed section.
- **Constant learning rate, optimizer state not saved:** add warmup and decay; resumed runs restart AdamW from scratch.
- **Unseeded split:** train/validation membership changes every run.
- **Sampling is on by default:** outputs vary between calls. Use `do_sample=False` or a lower temperature for consistent results.
- **Development server:** Flask's built-in server is not for production. Use gunicorn or waitress when deploying.
- **Not a medical device:** suggestions are general wellness prompts, not clinical advice.

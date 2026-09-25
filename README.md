# transformer-fundamentals

Build and train a small GPT from scratch in PyTorch, learn to debug a training
run from its metrics, find an induction head in a model you trained, then work
through ARENA's interpretability chapter.

**Goal:** debug a training run, and explain attention mechanics in an interview,
without notes.

The repo is stubs and tests. Every function you need to write raises
`NotImplementedError` and has a docstring spelling out its contract. The tests
check behaviour against PyTorch's own implementations (`nn.LayerNorm`,
`F.scaled_dot_product_attention`, autograd) and against properties you should be
able to state: causality, the step-0 loss being ln(V), the residual identity, and
so on. There are no solutions in the repo.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                 # everything fails; that's the starting line
```

A laptop CPU is enough for parts 1–4. ARENA needs a GPU (Colab is fine).

## Path

| Part | Where | You write | Exit criterion |
|---|---|---|---|
| 1. PyTorch fluency | `part1_pytorch/` | Broadcasting and einsum drills, backprop by hand, a bare training loop | `pytest tests/test_part1.py` green. You can derive ∂CE/∂logits and ∂L/∂W for a linear layer on paper. |
| 2. GPT from scratch | `part2_gpt/` | LayerNorm, embeddings, causal multi-head attention, MLP, block, GPT, param count, batching, loss, AdamW groups, lr schedule, the training loop, sampling | `pytest tests/test_model.py tests/test_train.py` green; `python -m part2_gpt.train` gets val loss ≤ 1.6 (about 7 min on a 4-core CPU), and samples look like Shakespeare. |
| 3. Debugging drills | `part3_debugging/` | Diagnoses: seven runs with one bug each, identified from metrics before reading the code | `DRILLS.md` table filled in, plus your own debugging checklist. |
| 4. Induction head | `part4_induction/` | Train a 2-layer attention-only model; score heads; ablate them. Includes a deliberate positional-shortcut trap. | You find the prev-token and induction heads, show that ablating the prev-token head breaks second-half loss, and can explain K-composition. |
| 5. ARENA chapter 1 | `arena/` | ARENA exercises in your own notebooks plus a half-page write-up per section | See `arena/README.md` for per-section exit criteria. |
| Throughout | `interview/QUESTIONS.md` | Timed, out-loud answers; misses logged in `MISSES.md` | All of section A under 3 min each, no notes. |

Run `pytest tests/test_model.py -x` while working on part 2. The tests are
ordered bottom-up, so the first failure is the next thing to build.

## Commands

```bash
python -m part1_pytorch.ex03_training_loop
python -m part2_gpt.train --name baseline          # writes runs/baseline/{metrics.csv,model.pt}
python -m part2_gpt.plot runs/baseline             # loss, grad norm, update ratio, lr
python -m part2_gpt.sample runs/baseline --prompt "ROMEO:" --top-k 20
python -m part3_debugging.drill_00_baseline        # then drill_01 .. drill_07; see DRILLS.md
python -m part4_induction.induction
```

## Rules for yourself

- **No copy-paste from nanoGPT, ARENA solutions, or an LLM** for parts 1–4.
  Look things up in the PyTorch docs; that's normal work. If you're stuck for more
  than 30 minutes, ask for a *hint*, not code.
- **Shapes in comments** on every tensor op the first time you write it.
- **Predict before you run.** Write down the step-0 loss, the loss at the end, and
  what the plot will look like. The gap between prediction and result is what you learn from.
- After each part, close the laptop and re-derive its core piece on paper.

## Reference numbers (char-level Tiny Shakespeare, V = 65)

| | loss (nats) |
|---|---|
| uniform, ln 65 | 4.17 |
| unigram entropy (character frequencies only) | ~3.3 |
| `drill_00_baseline` (64-d, 2 layers, 600 steps, <1 min) | ~2.05 val |
| `part2_gpt.train` default (128-d, 4 layers, 3000 steps, ~7 min) | ~1.5 val, ~1.25 train |

# ARENA: Transformer Interpretability (chapter 1)

Material: <https://arena.education> and <https://github.com/callummcdougall/ARENA_3.0>.
ARENA renumbers and adds sections over time, so this plan refers to sections by
**name**. Check the current table of contents before you start.

Run the notebooks on Colab or a rented GPU. Most chapter 1 sections load GPT-2
small or bigger models through TransformerLens, which is slow on a laptop CPU.

For each section, keep your own notebook in this folder (`arena/NN-name.ipynb`).
Write the exercises yourself, and only open the solutions after a real attempt.
Before moving on, write `arena/NN-name.md`: half a page answering the section's
"explain it" prompt below from memory.

## Order and exit criteria

| # | ARENA section | Why it's here | Done when you can explain, without notes… |
|---|---|---|---|
| 1 | **Transformer from scratch** | You've already built this in part 2 with the same weight conventions. Do it quickly as a check, and focus on loading GPT-2 weights into your code and matching its logits. | Where each GPT-2 weight goes in your model, and why the logits match to 1e-4. |
| 2 | **Intro to mech interp** (TransformerLens, induction heads) | Hooks, the activation cache, and induction heads in a real model. You've already found one in part 4. | QK and OV circuits; the full induction circuit as K-composition; what a hook point is and how you'd patch one. |
| 3 | **Indirect object identification** (circuit discovery) | The canonical circuit-finding case study: activation patching, path patching, name movers, S-inhibition. | The IOI circuit's head classes and the patching experiment that identified each one; what activation patching measures vs attribution patching. |
| 4 | **Superposition & SAEs** (toy models) | Why individual neurons aren't the right unit, and what sparse autoencoders do about it. | The toy-model phase diagram (sparsity × importance); why an SAE has an L1 penalty; what dead latents and feature splitting are. |
| 5 | **Interpretability with SAEs** | Using pretrained SAEs on a real model: feature dashboards, steering, SAE circuits. | How to go from an SAE latent to a claim about model behaviour, and what evidence would make you believe it. |
| 6 | Pick one: **Function vectors / steering**, **Probing**, **OthelloGPT**, **Grokking & modular arithmetic**, or **Balanced bracket classifier** | Depth in one applied technique. Grokking pairs well with the training-dynamics skills from part 3. | A 5-minute talk on the section's main result with one plot you made yourself. |

## Habits that make this stick

- Predict before you run. Before each plot, write one line saying what you expect it to show.
  The mismatches are where the learning is.
- Redo the key diagrams on paper a day later: the IOI circuit, the induction circuit,
  and the superposition phase diagram.
- When a result surprises you, go back to your part-2 model and reproduce it in
  miniature. The attention-only model from part 4 is a good testbed for
  activation patching.

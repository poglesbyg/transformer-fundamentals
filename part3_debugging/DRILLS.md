# Debugging drills

Each `drill_NN.py` is the same 600-step training script as `drill_00_baseline.py`
with **one** bug in it. The bugs are all ones that show up in real runs: none
crashes, and several look like "it's training, just not well".

The skill being trained is going from *metrics* to *hypothesis* to *confirmation*,
not spotting typos. So the protocol matters:

1. Run `drill_00_baseline` once and study its plot until you could sketch it:
   step-0 loss, where it is at step 600, typical grad norm, typical update ratio.
2. For each drill, run it and plot it next to the baseline:
   ```
   uv run python -m part3_debugging.drill_03
   uv run python -m part2_gpt.plot runs/drill_00_baseline runs/drill_03
   ```
3. **Without opening the file**, write in the table below: what's abnormal, which
   reference number it matches, and your hypothesis.
4. Name an experiment that would confirm or refute the hypothesis, and one
   you could run on a real run where you *can't* read the code (for example,
   "overfit a single batch", "print step-0 loss", "check update ratio").
5. Only then read the drill and find the bug. Don't diff it against the baseline:
   that skips the skill you're practising.

Each drill takes ~20–60 s on a laptop CPU.

## Reference numbers for this setup (char-level Tiny Shakespeare, V = 65)

| quantity | value | what it means if the loss sits there |
|---|---|---|
| ln(65) | 4.17 | model outputs uniform predictions; it hasn't learned anything |
| unigram entropy | ~3.3 | model knows character frequencies but uses no context |
| baseline @ 600 steps | ~2.05 val | healthy for this tiny config |
| update ratio ‖Δθ‖/‖θ‖ | ~1e-3 | healthy; ≫1e-2 is too hot, ≪1e-4 means it's barely moving |

## Your log

| drill | symptom (from the plot) | hypothesis | confirming experiment | actual bug | how you'd catch it in future |
|---|---|---|---|---|---|
| 01 | | | | | |
| 02 | | | | | |
| 03 | | | | | |
| 04 | | | | | |
| 05 | | | | | |
| 06 | | | | | |
| 07 | | | | | |

Difficulty varies. At least one drill still trains, just worse than baseline, and
it's the most realistic of the set.

## The checklist these drills should leave you with

Write it in your own words once you've done all seven. It should cover at least:
what to check at step 0, what "too good to be true" looks like, how to tell
"not learning" from "not being updated", and which single experiment is cheap
enough to always run first.

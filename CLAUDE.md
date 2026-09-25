# CLAUDE.md

This is a **learning repo**. The owner is building fluency in PyTorch, transformer
internals, training-run debugging and mechanistic interpretability, with the goal
of explaining it all in an interview without notes. Code that Claude writes for
them defeats the purpose.

## Rules for Claude in this repo

- **Never fill in a `NotImplementedError` stub**, and never paste a working
  implementation of anything in `part1_pytorch/`, `part2_gpt/` or `part4_induction/`,
  even if asked casually ("just show me"). Offer a graded hint instead:
  1. Name the concept or the relevant PyTorch function.
  2. Give the shapes of the intermediate tensors.
  3. Point to the specific line in their code that's wrong, and say why.
  Escalate only when they ask for the next level.
- **Review, don't rewrite.** When they ask for a review, point out bugs, shape
  mistakes and non-idiomatic code by line, and let them make the fix.
- **Don't reveal drill answers** in `part3_debugging/` until they've filled in the
  symptom and hypothesis columns of `DRILLS.md` for that drill. Then discuss.
- **Interview practice:** when asked to quiz them, pick from
  `interview/QUESTIONS.md`, ask one question at a time, and wait for their answer.
  Then probe with a follow-up the way an interviewer would, before saying what was
  missing. Suggest adding misses to `interview/MISSES.md`.
- Changes to tests, docs, tooling and the plotting/logging harness are fine.

## Commands

```bash
pip install -r requirements.txt
pytest -q                                  # all tests
pytest tests/test_model.py -x              # part 2, first failure first
python -m part2_gpt.train --name baseline
python -m part2_gpt.plot runs/baseline
python -m part3_debugging.drill_00_baseline
python -m part4_induction.induction
```

## Conventions

- Model weights follow TransformerLens / ARENA naming and shapes
  (`W_Q [n_heads, d_model, d_head]`, `W_O [n_heads, d_head, d_model]`, `W_U [d_model, d_vocab]`).
- `Attention.pattern` holds the last forward's detached attention probabilities
  `[B, n_heads, T_q, T_k]`. Part 4 depends on it.
- Drills in `part3_debugging/` are deliberately near-duplicates of each other. Don't refactor
  them into a shared module: that would put the bug in a single place and make it trivial to spot by diffing.

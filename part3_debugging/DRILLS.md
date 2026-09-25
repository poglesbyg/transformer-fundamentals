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
| 01 
What the metrics show

step	baseline val	drill 01 val	baseline update ratio	drill 01 update ratio
150	2.43	2.66	7.6e-3	1.5e-2
599	2.08	2.42	8.3e-4	1.5e-3
Loss: it does learn. It gets well past the character-frequency plateau at 3.3, so the model, data and loss are fine. It just learns worse, ending about 0.35 above the baseline.
Update ratio: about 2× the baseline throughout. Each step moves the weights further than it should for the same learning rate.
Gradient norm: slightly higher (1.2–1.4 vs about 1.1). It's elevated but not exploding.

The reasoning chain:

It learns, so it's not a data, loss or wiring bug. Those give flat or suspicious loss curves: drills 02, 04, 05, 06.
The weights move further per step, and the learning rate is identical, so the gradient being applied must be bigger or staler than normal.
Gradients that are "too large but somehow not exploding" point to something that accumulates and is then capped.

The bug: opt.zero_grad() was deleted. PyTorch adds each new gradient into .grad, so without zeroing, every step's gradient is the sum of the current gradient and all previous ones.

Why it doesn't blow up: clip_grad_norm_ rescales .grad in place to norm 1.0, so the running sum is capped every step. The result is a gradient that is mostly stale history plus a little of the current batch. It behaves like very heavy momentum stacked on top of Adam's own momentum, which pushes the weights in outdated directions. Without clipping, the gradient norm would grow roughly linearly with the step count, and you'd have spotted it instantly.

How you'd catch this on a real run (the column to fill in DRILLS.md):

Turn off clipping for a few steps and watch the gradient norm. Linear growth means accumulation.
Add a check at the top of each step: assert all(p.grad is None for p in model.parameters()). That holds if you zero with set_to_none=True.
Watch for an update ratio that's consistently higher than a known-good run at the same learning rate.

The general lesson: gradient clipping can hide bugs. A run that "trains, just a bit worse" with clipping on deserves a look at the raw gradient norm with clipping off.

| 02 | | | | | |

What the metrics show (from my run):

step	train	val	grad norm
0	4.17	4.14	2.7
150	0.002	0.002	0.006
599	0.000	0.000	0.000

The reasoning chain:

Loss 0.000 is impossible on this task. Predicting the next character of English has irreducible uncertainty: after "th", the next character could be e, a, i, o, r and so on. A good character model bottoms out well above 1 nat, and your baseline stopped around 1.5. So near-zero loss means the model is being handed the answer.
Validation is also zero, so this isn't overfitting. Overfitting shows training loss far below validation loss. Both at zero means the task itself is broken, on every split.
It collapses in about 100 steps, faster than any real learning happens. Whatever the model learned is trivial.
The gradient norm drops to 0. The model is perfectly confident and has nothing left to learn.

So the question becomes: what trivial function gets zero loss? Answer: copying the input.

The bug: in batch(), the targets aren't shifted.

y = torch.stack([data[i : i + T] for i in ix])       # bug: y == x
y = torch.stack([data[i + 1 : i + 1 + T] for i in ix])  # correct

With y == x, the target at position t is the token at position t. Causal attention lets position t see itself, so the model learns to copy the current token through the embedding and unembedding, and the loss goes to zero. Note that the causal mask can't catch this: it's correct, and it's the data that's wrong.

How you'd catch it on a real run:

Know the loss floor for your task. A loss below what's achievable is a bug, not a breakthrough. Write down your expected range before every run.
Sample from the model. It will just emit its prompt back or produce garbage, which contradicts the "perfect" loss.
Check the batch directly: assert (y[:, :-1] == x[:, 1:]).all(). That's what test_get_batch_shift_and_bounds in your part 2 tests checks.

| 03

What the metrics show (from my run):

step	train	grad norm	update ratio
0	17.7	16.8	6e-5
150	3.29	0.44	5e-4
599	3.02	0.39	7e-5

Baseline for comparison: step 0 at 4.17, step 599 at 2.02, update ratio around 1e-3.

The reasoning chain:

Step 0 is 17.7, where it should be ln 65 = 4.17. An untrained model should be uncertain, with roughly uniform predictions. A loss of 17.7 means it's confidently wrong: it puts nearly all its probability on essentially random tokens. The model hasn't learned anything yet, so this must be the weights it started with.
It drops fast, to about 3.3, the character-frequency entropy, and then stalls there. It learns how common each character is, but barely gets any context-dependent structure beyond that.
The update ratio is 10–50× smaller than the baseline. Adam moves each parameter by roughly the learning rate per step no matter how big the weights are. If the weights are 50× larger, each step changes them 50× less relative to their size. So the model is effectively frozen in its bad starting point.

Huge initial loss plus a tiny update ratio adds up to one explanation: the weights started far too large.

The bug: in init_weights, std=0.02 became std=1.0.

Why the loss is huge: LayerNorm outputs have roughly unit variance, so logits = LN(x) @ W_U have a standard deviation of about √d_model = 8. That makes the softmax extremely peaked on arbitrary tokens, and cross-entropy punishes confident mistakes severely.
Why it stalls: attention scores are equally large, so the attention softmaxes are saturated one-hot patterns with near-zero gradients through them. On top of that, Adam's fixed-size steps barely move the large weights.

How you'd catch it on a real run:

Always check the step-0 loss against ln(V). That's the single cheapest sanity check in language modelling, and it's what test_gpt_init_loss_is_uniform checked in part 2. Being within about 0.1 of ln(V) means initialization and the forward pass are sane. Far above it means logits are too big: check the init scale, or a missing LayerNorm before the unembedding.
Check the update ratio. Far below 1e-3 from the start means the weights are too large relative to the learning rate.

| 04 

What the metrics show (from my run):

step	train	val	grad norm
0	4.17	4.16	1.76
150	3.32	3.39	0.17
599	3.31	3.34	0.17

The reasoning chain:

Step 0 is 4.17, which is correct. So initialization and the forward pass are fine, unlike drill 03.
The loss drops and then stalls at 3.31. That's exactly the character-frequency entropy from the reference table: the loss you get by knowing how common each character is and ignoring context entirely. A value this exact is a clue, not a coincidence.
The gradient norm settles low, around 0.17. The model has found everything it can learn and has nothing more to extract.
Train and val sit together at 3.3. Nothing is overfitting. The task as presented to the model just contains no usable signal beyond character frequencies.

So the question becomes: what makes the context useless? It happens when the targets are no longer related to the inputs they're paired with, while still being real characters with the right overall frequencies.

The bug: in loss_fn, the targets are flattened in a different order from the logits:

F.cross_entropy(logits.reshape(-1, V), y.t().reshape(-1))   # bug
F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))       # correct

y is [B, T] = [32, 64]. y.t() is [64, 32], and flattening that produces the targets in time-major order while the logits are in batch-major order. So the logit for (sequence b, position t) is scored against the target for some other sequence and position. The element count matches, so nothing crashes. And because every target is still a real character, the best the model can do is predict the overall character distribution, which is exactly 3.31.

The neat confirmation experiment: rerun with batch size 1. For a [1, T] tensor, y.t().reshape(-1) gives the same order as y.reshape(-1), so the bug disappears and the loss drops normally. When a bug vanishes at B = 1, suspect the batch and sequence axes are being mixed up.

How you'd catch it on a real run:

Know your plateau numbers. Stalling at the unigram entropy means "the model can't use context". Then look for anything that breaks the input–target pairing: reshapes, transposes, shuffles, off-by-one shifts.
Be suspicious of any flattening with a transpose. reshape, view and flatten never check that axes line up, only that sizes match. That's the same view-and-copy territory as your lm_loss bug in part 2.
Check alignment directly: make sure x[b, t+1] == y[b, t] survives the whole path to the loss.

| 05 

What the metrics show (from my run):

step	train	grad norm	update ratio
0	4.17	0.029	1e-3
150	3.95	0.12	6e-3
599	3.94	0.14	9e-4

Baseline for comparison: step-0 gradient norm 1.83, and the loss is at 2.02 by step 599.

The reasoning chain:

Step 0 is 4.17, which is correct. So initialization and the forward pass are fine.
The step-0 gradient norm is 0.029, about 60× smaller than the baseline with the same model and data. Something between the logits and the loss is squashing the gradient.
The loss stalls at 3.94, above the unigram entropy of 3.31. It can't even learn character frequencies, which even drill 04's broken setup managed. So the loss function itself is limiting how low the loss can go, whatever the model does.
The update ratio looks normal, because Adam rescales updates to a normal size. That's why "update ratio fine but loss not moving" points at the loss path, not the optimizer.

The bug: softmax is applied before cross-entropy.

F.cross_entropy(logits.softmax(-1).reshape(-1, V), y.reshape(-1))   # bug
F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))               # correct

F.cross_entropy expects raw logits and applies log_softmax internally. Given probabilities instead, it computes a softmax of probabilities. Every input is in [0, 1], so the best possible case is a one-hot [1, 0, …, 0], and even that only gives the correct class e / (e + 64) ≈ 4%. The loss can never go below ln(1 + 64/e) ≈ 3.2, no matter how good the model is. On top of that, the gradient passes through two softmaxes, and the first one's Jacobian shrinks it, which is why the gradient norm is tiny.

How you'd catch it on a real run:

Compare the step-0 gradient norm to a known-good run. A healthy step-0 loss with a gradient orders of magnitude smaller means something is squashing the signal on its way to the loss.
Know what your loss function expects. CrossEntropyLoss is LogSoftmax plus NLLLoss, and it wants raw logits. The same mistake appears as sigmoid before BCEWithLogitsLoss.
Compute a floor from first principles. If the loss can't beat the unigram entropy, check whether the pipeline makes it mathematically impossible.

| 06 

The single most informative number is an exact zero.

What the metrics show (from my run):

step	train	val	grad norm	update ratio
0	4.16	4.17	1.83	0.0
150	4.17	4.17	2.96	0.0
599	4.17	4.17	2.95	0.0

The reasoning chain:

Step 0 is 4.17, which is correct, and the loss never moves from ln(V). The model learns nothing at all, not even character frequencies, unlike drills 04 and 05.
The gradient norm is healthy and nonzero. So the forward pass, the loss and backprop all work, and gradients do arrive on the model's parameters.
The update ratio is exactly 0.0. Not small, exactly zero. The parameters of the model being measured don't change at all between steps.

Gradients arrive but the weights never change. So optimizer.step() runs, but on something other than these parameters.

The bug: the optimizer is attached to a different model object.

model = GPT(cfg)
opt = torch.optim.AdamW(model.parameters(), ...)   # optimizer holds model #1's params
torch.manual_seed(0)  # re-seed so every drill starts from identical weights
model = GPT(cfg)                                   # name now points to model #2
init_weights(model)

The loop does forward and backward on model #2, so model #2 gets gradients, which is why the gradient norm is nonzero. But opt.step() updates model #1, which is no longer used anywhere. Model #2 never moves. The comment makes the line look deliberate, and in real code it usually is. This happens with "reset for reproducibility", loading a checkpoint after creating the optimizer, model = model.to(device) in some frameworks, or wrapping in DDP or torch.compile after the optimizer exists.

(The gradient norm creeping from 1.8 up to 2.9 comes from the learning-rate warmup: the gradient is measured on a model that's frozen but seeing different batches, so it just fluctuates around that fixed point.)

How you'd catch it on a real run:

Log the update ratio. Exactly 0 while the gradient norm is nonzero means the optimizer and the model are disconnected. It's the cheapest possible detector, and it's why train() logs it.
Build the optimizer last, after every step that might replace the model: loading, moving to a device, wrapping, compiling.
Assert the link: check that {id(p) for g in opt.param_groups for p in g["params"]} == {id(p) for p in model.parameters()}.

| 07

t gives itself away in the step-0 row.

What the metrics show (from my run):

step	train	val	grad norm	update ratio
0	4.16	3.69	1.83	1.8
150	3.13	3.13	0.18	1.8e-2
599	2.80	2.80	0.30	1.7e-3

Baseline for comparison: step-0 update ratio 1e-3, final validation loss 2.08.

The reasoning chain:

Step 0 is 4.16, which is correct. Initialization and the forward pass are fine.
The step-0 update ratio is 1.8. A single optimizer step changed the weights by 180% of their own size. The healthy range is about 1e-3. In one step the carefully scaled initialization was wiped out.
Next comes a long stretch near the unigram plateau with a small gradient norm (0.18). That's the same fingerprint as drill 03: weights that are far too large, saturated softmaxes, small gradients. The difference is that here the optimizer made the weights large, rather than the initialization.
It slowly recovers as the learning rate decays and finishes at 2.80, which is still far behind the baseline.

The bug: the learning rate is 30× too high, with no warmup.

LR = 1e-1       # baseline 3e-3
WARMUP = 1      # baseline 50

Adam moves each parameter by roughly the learning rate per step, whatever the gradient's size. The initial weights are around 0.02 in magnitude, so a step of about 0.1 is five times their own size. Warmup exists for exactly this reason: Adam's running estimates of gradient size are poor in the first few steps, and small early steps give the model time to settle before full-size updates start.

How you'd catch it on a real run:

Check the update ratio. Anything above about 1e-2 is too aggressive, and above 1 the weights are effectively being overwritten every step.
Look for loss spikes, or a stall right after warmup ends.
Do a quick learning-rate sweep. Run 100 steps each at a few learning rates (1e-4, 3e-4, 1e-3, 3e-3, 1e-2) and pick the largest one whose loss still drops smoothly. It's cheap and removes the guesswork.


Difficulty varies. At least one drill still trains, just worse than baseline, and
it's the most realistic of the set.

## The checklist these drills should leave you with

Write it in your own words once you've done all seven. It should cover at least:
what to check at step 0, what "too good to be true" looks like, how to tell
"not learning" from "not being updated", and which single experiment is cheap
enough to always run first.

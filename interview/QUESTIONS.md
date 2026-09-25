# Interview drill

There are no answers in this repo on purpose. Answer out loud, timed, on a
whiteboard or blank paper. Then check against your own code and notes and write
down only what you got wrong in `interview/MISSES.md`. Retest the misses a few
days later.

"Done" means you can do every question in section A in under 3 minutes each
with no notes, and handle a follow-up on any of them.

## A. Attention mechanics

1. Write single-head causal self-attention from scratch, with shapes on every line.
2. Why divide by √d_head? What happens to the softmax without it as d_head grows?
   Give the variance argument.
3. Where is the causal mask applied and with what value? What goes wrong if you
   mask *after* the softmax?
4. What does multi-head attention buy over one head with the same total width?
   Where do heads get combined?
5. Rewrite attention in terms of W_QK = W_Q W_Kᵀ and W_OV = W_V W_O. What does each
   matrix "do", and what is its rank?
6. Cost of one attention layer in FLOPs and memory as a function of T and d.
   Which term dominates at long context?
7. What is a KV cache? Compute its size in bytes for GPT-2 small (12 layers,
   d = 768) at 1024 tokens in fp16. What changes with multi-query or grouped-query attention?
8. Learned absolute positions vs RoPE: what does each add, and where? What
   does RoPE make attention scores depend on?
9. Pre-LN vs post-LN: where does the LayerNorm sit, and why is pre-LN easier to train deep?
10. What is the residual stream? Why can you treat each layer as reading from and
    writing to it?
11. Explain an induction head: the two heads, which layer each sits in, what
    each attends to, and why it needs two layers (K-composition).
12. What does FlashAttention change, and what does it leave unchanged?

## B. The model as a whole

13. Parameter count of GPT-2 small from the config. Where do most parameters live?
14. Training FLOPs ≈ 6·N·D. Where do the 6 and the 2 (forward) come from?
15. Why is the step-0 loss of a well-initialised LM ≈ ln(V)? What init makes that true?
16. Weight tying between embedding and unembedding: what it saves, and what it assumes.
17. Why AdamW over Adam + L2? Why no weight decay on LayerNorm gains and biases?
18. Why warmup? What does the loss curve look like without it at a high lr?

## C. Debugging a training run

For each, say what you'd check first and what result would confirm your hypothesis.

19. The loss is 17 at step 0 with V = 65.
20. The loss drops to ~0.00 within 150 steps on natural text.
21. The loss plateaus at the unigram entropy.
22. The loss doesn't move from ln(V), but grad norm is non-zero.
23. The loss goes NaN at step 2,000 after looking healthy.
24. Train loss keeps falling while val loss rises from step 3,000.
25. It trains, but noticeably worse than last week's run with "the same" config.
26. Throughput halved after a refactor. How do you find where the time went?
27. What's the single cheapest sanity check you always run first, and what does
    passing it rule out?

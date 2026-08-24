# Method: ECHO-CAVER

## 1. Problem

FREB-CAVER injects query-reference comparison evidence into selected JUDO
decoder blocks. A frozen downstream decoder must nevertheless convert that new
state into task-specific answer logits. The working hypothesis behind ECHO is
that some FREB gains remain unread: generic downstream LoRA can learn a useful
answer mapping, but can also create unrelated language directions that disturb
reasoning and previously correct tasks.

ECHO addresses this writer-reader mismatch by adapting only the blocks
immediately downstream of FREB's evidence writers and restricting the learned
output directions to the carrier's own evidence span.

## 2. Writer-reader alignment

Let FREB writer `i` inject a residual through its frozen matrix
`W_i in R^(d x h)` at decoder layer `l_i`. Its compact singular value
decomposition is

```text
W_i = U_i Sigma_i V_i^T.
```

ECHO retains the first `k=32` columns of `U_i` and defines the evidence
projector

```text
P_i = U_i U_i^T.
```

A rank-16 LoRA reader is installed in both the attention output projection and
MLP down projection of layer `l_i+1`:

```text
y = W_0 x + (alpha/r) B_i A_i x,
r = 16, alpha = 32.
```

The LoRA-B gradient is projected through `P_i`; after every AdamW update, the
weight is projected again:

```text
grad(B_i) <- P_i grad(B_i),
B_i       <- P_i B_i.
```

This two-stage enforcement prevents optimizer state from accumulating an
off-span component. The reader layers are `19, 21, 23, 25, 27`, paired with
FREB writer layers `18, 20, 22, 24, 26`. Ten modules are adapted in total.

## 3. Why this differs from ordinary and orthogonal LoRA

Ordinary LoRA can write in any learned low-rank output direction. O-LoRA-style
continual-learning methods separate tasks by training a new adapter orthogonal
to previously learned task subspaces. ECHO makes the opposite geometric choice:
the new reader is *tangent to* the existing causal evidence carrier. Its purpose
is not to isolate a new task, but to improve interpretation of a frozen visual
state without opening an unconstrained language-only residual channel.

The contribution is the coupling of:

1. causal writer-reader placement derived from the actual FREB injection sites;
2. singular-span constraints derived from the frozen writer weights;
3. gradient and post-optimizer weight projection;
4. exact zero-initialized identity with the deployed FREB computation;
5. answer repair, semantic option-orbit equivariance, and frozen-FREB retention.

## 4. Training objective

Each training item contains an original multiple-choice question and a semantic
option permutation. For candidate logits `z`, frozen FREB logits `z0`, target
`y`, and the inverse semantic permutation `pi`, the implementation optimizes

```text
L = L_CE
  + 0.25 L_equiv(z_original, pi^-1(z_permuted))
  + 1.00 KL(p0 || p)
  + 0.75 L_repair(margin=0.25).
```

`L_repair` is active when frozen FREB ranks an incorrect answer above the gold
answer; it asks the candidate to reverse that ordering by the specified margin.
The training set contains 963 non-AD pairs from the CAPRI-CAVER panel, with 184
validation pairs. ECHO v1 was trained for 100 steps with batch size two pairs,
AdamW, learning rate `2e-5`, zero weight decay, and seed `20260824`.

## 5. Parameter and runtime contract

| Item | Value |
|---|---:|
| Base parameters | 8,327,619,987 |
| Trainable ECHO parameters | 2,375,680 |
| Trainable fraction | 0.02853% |
| ECHO checkpoint size | 9.1MB |
| Training time | 183.07s on one H200 |
| Peak training allocation | 23.84GiB |

The mechanics receipt establishes zero-init exact identity (`0.0` maximum
absolute difference), nonzero first-step LoRA-B gradients at all ten targets,
and rank-32 projection of every LoRA-B matrix.

At inference, the public JUDO checkpoint, released FREB-CAVER adapter, and ECHO
PEFT adapter are loaded into one model runtime. The generated completion is the
only prediction source. No score threshold or external selection policy is
part of the method.

## 6. Known failure mode

ECHO changed the free-generated reasoning trace on 1,349/5,600 clean-panel
examples. This shows that evidence-span restriction alone does not localize the
adapter to the final answer state. The format-invalid count also increased in
the full census. A prospective successor should share FREB's causal decision
mask with ECHO readers so that segmentation and reasoning tokens remain the
frozen FREB computation while answer-state reading is adapted.


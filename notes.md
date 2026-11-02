# CS336 Notes

## 1. Tokenizer

A tokenizer maps **language → integers** (and back), so that raw text can be fed to a model.

### Three typical granularities

| Level | Vocab size | Trade-off |
|-------|-----------|-----------|
| **Character level** | ~26 for English | Tiny vocab, but sequences become very long |
| **Subword level** | < 100k (modern: 100–200k) | Data-driven "mini model"; good balance; works for multi-modal |
| **Word level** | Very large / sparse | Suffers from out-of-vocabulary (unknown) tokens |

Subword tokenization is the modern default: it keeps the vocabulary bounded while keeping sequences reasonably short, and gracefully handles rare words by splitting them into pieces.

### Pipeline

1. **Pre-tokenization**
   - Handle **special tokens** first (e.g. end-of-document `<|endoftext|>`).
   - Split text with a **regular expression** into coarse pieces.
   - **Chunk** the corpus and compute **word (pre-token) frequencies**.
2. **BPE (Byte-Pair Encoding) training**
   - Count the frequency of adjacent token **pairs**.
   - Repeatedly **merge** the most frequent pair into a new token, until the target vocab size is reached.

---

## 2. Einops & Compute/Memory Accounting

`einops` is available in PyTorch and makes tensor reshapes/reductions explicit and readable.

### Floating-point data types

Common training dtypes — **FP32, FP16, BF16** — differ in how many bits go to the **sign**, **exponent**, and **fraction (mantissa)**:

- *Exponent bits* → dynamic range (how large/small numbers can be).
- *Fraction bits* → precision (how finely numbers are resolved).

For a 16-bit float with a 5-bit exponent (bias 15) and 10-bit fraction, the value is:

$$
x = (-1)^s \times 2^{E-15} \times \left(1 + \frac{F}{2^{10}}\right)
$$

BF16 keeps FP32's exponent range (fewer fraction bits), which is why it's preferred for training stability.

### Mixed-precision training

- **Forward/backward pass:** data, parameters, activations, and gradients in **BF16**.
- **Optimizer states:** kept in **FP32** (the moving averages of the gradient and of the squared gradient).
- **Update step:**
  1. Update the master FP32 parameters using the FP32 optimizer states.
  2. **Down-cast** the updated weights back to BF16 for the next forward pass.

Keeping a high-precision copy for the update avoids losing small gradient contributions.

### Performance model

$$
\text{time} \approx \max(\text{computation}, \text{communication})
$$

A workload is either:

- **Compute bound** — limited by FLOP throughput, or
- **Memory bound** — limited by moving bytes to/from memory.

Two related notions of **intensity** (FLOPs per byte):

- **Accelerator intensity** — the hardware's ratio, set by the GPU and the data type.
- **Arithmetic intensity** — the workload's ratio, set by the model/architecture and the data type.

If arithmetic intensity < accelerator intensity → you're memory bound; otherwise compute bound.

### Memory-saving techniques

- **Gradient accumulation** — simulate a large batch with small per-step compute/memory. Memory scales linearly with the (mini-)batch size. Split a large batch into *K* minibatches; for each minibatch you store/compute weights, activations, data, gradients, and gradient-squares, then accumulate.
- **Activation checkpointing** — trade compute for memory by saving activations for only some layers and **recomputing** the rest during the backward pass.

---

## 3. Architecture

Starting point: the traditional Transformer, with the following modern refinements.

### 1. Normalization placement
- **Pre-norm dominates** over post-norm: it lets gradients flow directly through deep stacks (like residual networks), improving training stability.
- **RMSNorm vs. LayerNorm:** RMSNorm is cheaper (no mean-centering / bias term) with little-to-no loss in quality.

### 2. Positional encoding
- **RoPE (Rotary Positional Encoding)** dominates modern models.
- It is **relative** and position-invariant: the score depends only on the *offset* between positions — `f(x, y, i, j) = g(x, y, i - j)`.
- Applied to the **queries and keys** inside the attention computation, **not** to the values.

### 3. Activation function
- Choice (**gated vs. non-gated**, e.g. SwiGLU vs. ReLU/GeLU) is largely empirical; gated variants tend to help.

### 4. Hyperparameters
- `d_ffn` vs. `d_model`: a factor of **~4** is the rule of thumb.
- Embedding dim: `d_model = num_heads × d_head`.

### 5. Aspect ratio (width vs. depth)
- `d_model / num_layers` is typically in the **100–200** range (up to several hundred).
- **Deeper models are harder to train** than wider models (optimization and stability get worse with depth).

### 6. Vocabulary size
- Roughly **100–300k** for modern models.

### 7. Regularization & training stability
- **Weight decay** is the main regularizer; **dropout is no longer popular** for LLM pretraining.
- **Watch out for softmaxes** — they are a common source of instability:
  - **z-loss** — penalize the magnitude of the logits to keep them bounded.
  - **Add normalization** (e.g. RMSNorm) in the right places to stabilize.

### 8. Attention interventions
- **Multi-Query Attention (MQA):** a single shared head for **K** and **V** (many query heads).
- **Grouped-Query Attention (GQA):** several K/V heads — more than 1 (MQA) but fewer than full MHA.
- **Mixing full and sparse/"light" attention layers** to reduce cost.

---

## 4. Variations of Attention & MoE

*(to be filled in)*

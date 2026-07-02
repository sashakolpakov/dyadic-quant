# Progressive Dyadic Quantization for Memory-Adaptive Neural Inference

## Abstract

Conventional post-training quantization produces a separate model for each
target precision. A four-bit model is not generally a prefix of an eight-bit
model, so increasing the available memory requires replacing or re-encoding
the weights. This paper studies a progressive alternative: encode each weight
once as an MSB-first signed dyadic code, then execute any selected prefix of
that code. Adding one precision level transfers exactly one additional bit per
encoded weight and leaves all previously stored planes unchanged.

The construction uses power-of-two exponents, midpoint decoding of dyadic
intervals, optional clipping of outliers, and a normalized-regret criterion that
chooses each shared exponent jointly for several prefix widths. We evaluate this
representation on ResNet-18 and a 494M parameter Qwen2.5 model on an Apple M5
GPU. Six-bit dyadic convolution weights come within 0.99 percentage points of
ResNet-18 Imagenette FP16 accuracy while reducing estimated stored model size
from 23.37 MB to 9.43 MB; eight-bit weights come within 0.08 points. On
Qwen2.5, moving from one exponent per output channel to one exponent per
32-weight group rescues the low-bit regime while preserving progressive
refinement: at 6.25 effective bits/weight, group-32 dyadic reaches WikiText-2
perplexity 28.88 and 100% ARC-Easy textual meaning equivalence on a 20-prompt
sample, comparable to or better than the independently built GGUF controls at
similar effective sizes. These results show that one nested code can support
several useful memory-quality operating points. They do not yet demonstrate
low-bit inference acceleration, because the current MPS implementation
materializes every prefix as FP16. A packed Metal or llama.cpp kernel remains
necessary to test the anticipated speed and energy benefits.

## 1. Introduction

Quantization is normally treated as a choice made before deployment: a model
is converted to INT4, INT8, or another fixed representation, and the selected
format determines its memory and accuracy. This is effective when the target
device and memory budget are known in advance, but it is less suitable when
the available memory changes dynamically or when one artifact must serve
several devices.

The objective studied here is stronger than ordinary low-bit quantization. We
seek one encoded model with the following properties:

1. its first few bit planes define a valid low-memory model;
2. additional planes refine that same model without replacing earlier data;
3. every refinement costs one bit per encoded weight;
4. decoding uses simple binary operations and power-of-two scaling;
5. useful model quality is retained at practical prefix widths.

The motivating mathematical structure is dyadic, or equivalently
two-adic at the level of digit organization. In an ordinary two-adic integer,
digits are accumulated in increasing powers of two. To obtain progressively
finer real approximations, their significance must instead be interpreted in
the real direction, as in the Monna digit-reversal map [1]. The resulting
representation is not a ring embedding from the two-adics into the reals:
native two-adic addition and multiplication do not become ordinary real
arithmetic. Its practical value lies elsewhere. The two-adic digit hierarchy
provides a nested storage order, while inference uses fixed-point or
bit-plane arithmetic after decoding.

This paper develops such a representation and evaluates it on convolutional
and transformer models. The main empirical finding is that six-bit prefixes
are a useful operating point in both cases. The main engineering limitation
is that the present implementation validates numerical quality and storage,
not packed execution speed.

## 2. Progressive dyadic representation

### 2.1 One maximum-depth code, many prefixes

Consider a weight tensor whose first dimension indexes output channels. For
channel $c$, choose an integer exponent $e_c$ and define the finest dyadic
step


```math
\Delta_c = 2^{e_c}.
```


Let $B$ be the maximum stored width. Each weight is represented logically by
one sign and a nonnegative magnitude code $q_{c,i}^{(B)}$. At maximum depth,
the midpoint reconstruction is


```math
\widehat w_{c,i}^{(B)} = s_{c,i}\Delta_c \left(q_{c,i}^{(B)}+\frac12\right), \qquad s_{c,i}\in\{-1,+1\}.
```


For a shorter $b$-bit prefix, lower magnitude planes are discarded:


```math
q_{c,i}^{(b)} = q_{c,i}^{(B)} \gg (B-b).
```


The effective step grows accordingly:


```math
\Delta_c^{(b)} = 2^{e_c+B-b}.
```


The corresponding reconstruction is


```math
\widehat w_{c,i}^{(b)} = s_{c,i}\Delta_c^{(b)} \left(q_{c,i}^{(b)}+\frac12\right).
```


Thus the same stored code provides all requested precisions. Moving from
$b$ to $b+1$ reveals one additional magnitude plane and divides every
current dyadic interval into two halves.

The following diagram shows one weight stored at maximum depth $B=8$ (one
sign plane plus seven MSB-first magnitude planes), and how reading a prefix is
simply dropping the low planes:

```
 one stored max-depth code (B = 8):   sign  m6 m5 m4 m3 m2 m1 m0   (MSB-first)
                                      ┌────┬──┬──┬──┬──┬──┬──┬──┐
                                      │ s  │ 1│ 0│ 1│ 1│ 0│ 0│ 1│
                                      └────┴──┴──┴──┴──┴──┴──┴──┘
                                            └──────── magnitude ───────┘

   read 4-bit prefix  ->  s | 1 0 1            (q >> 4 ; step = 2^(e+4))
   read 6-bit prefix  ->  s | 1 0 1 1 0        (q >> 2 ; step = 2^(e+2))
   read 8-bit (full)  ->  s | 1 0 1 1 0 0 1    (q       ; step = 2^e    )

   value(b) = sign · 2^(e + B - b) · (prefix_code + 0.5)

   refinement: adding one bit of precision = appending the next stored
   plane; every earlier plane and the exponent e are unchanged.
```

The midpoint term is important. A prefix identifies an interval of possible
fine-resolution values. Decoding at the lower edge systematically reduces
weight magnitudes. In a deep model, this bias compounds across layers.
Midpoint decoding approximately centers the truncation error and requires no
additional per-weight metadata.

### 2.2 Storage and refinement

With packed storage, a $b$-bit model requires approximately


```math
\frac{bN}{8}
```


bytes for $N$ encoded weights, plus one small exponent per output channel
and any parameters intentionally left at higher precision. Increasing the
prefix width by one requires exactly


```math
\left\lceil\frac{N}{8}\right\rceil
```


additional bytes.

This differs from maintaining separate INT4, INT6, and INT8 artifacts.
Independent quantizers may achieve lower error at each individual width, but
their codes and scales are generally unrelated. They do not provide an
incremental transfer or an in-place refinement guarantee.

### 2.3 Why power-of-two scales?

An unconstrained floating-point scale can be selected more accurately, but a
power-of-two scale has two advantages:

- it can be represented by a small signed exponent;
- in fixed-point hardware, applying it reduces to a shift rather than a
  general multiplication.

The constraint is not free. If the exponent is selected only to cover the
largest channel weight, most of the code range may be wasted on an outlier.
The implementation therefore evaluates nearby exponents and permits clipping
when clipping lowers channel reconstruction error.

Per-channel rather than per-layer exponents are necessary because output
channels can have substantially different ranges. A single layer scale causes
small-range channels to spend their leading planes on zeros, delaying useful
information until later prefixes.

### 2.4 Scale granularity: the group size

The exponent $e$ is shared by a *group* of weights. It must be large enough
that the largest-magnitude weight in the group is representable; but a step
$2^e$ large enough for an outlier is too coarse for the small weights that
share it, which then snap to a sparse grid. The number of weights sharing one
exponent — the **group size** $g$ — therefore directly controls the
range-versus-resolution conflict.

Per-channel scaling is the special case $g = K$, where $K$ is the number of
weights in an output row. Reducing $g$ gives each small block its own
exponent, so a single large weight only coarsens its own block instead of an
entire row. This is the same locality that block quantizers such as the
llama.cpp k-quants exploit with 16–32-element blocks.

```
 one output row of K weights, grouped with size g:

   w0 w1 … w(g-1) │ wg … w(2g-1) │ w2g … │ …
   └─── group 0 ──┘ └── group 1 ──┘ └ grp 2┘
        2^e0             2^e1          2^e2     one exponent per group

   g = K      (per-channel) : 1 exponent / row        + 0      bpw
   g = 64     (block-wise)  : K/64 exponents / row     + 0.125  bpw
   g = 32                   : K/32 exponents / row      + 0.25   bpw
   g = 16                   : K/16 exponents / row      + 0.5    bpw
```

Smaller groups improve fidelity — especially at low prefix widths, where a
coarse step is most damaging — at the cost of one extra exponent byte per
group. The magnitude planes themselves are unchanged, so block-wise exponents
preserve both the progressive-prefix property and the shift-only decode. The
group size is thus a tunable knob between storage overhead and accuracy, swept
empirically in Section 6.

## 3. Choosing one exponent for several precisions

### 3.1 The scale-conflict problem

For a candidate exponent $e$, define the channel reconstruction MSE at
prefix width $b$:


```math
E_b(e) = \frac{1}{n_c} \sum_i \left( w_{c,i}-\widehat w_{c,i}^{(b)}(e) \right)^2.
```


If each width were encoded independently, it could use


```math
e_b^* = \arg\min_e E_b(e).
```


A progressive model cannot do this: all widths must share one exponent and
one maximum-depth code. The first implementation minimized raw summed error,


```math
\arg\min_e \sum_{b\in\mathcal B}E_b(e),
```


for $\mathcal B=\{4,5,6,8\}$. This proved unsuitable because low-bit errors
are naturally much larger than high-bit errors. Four-bit MSE dominated the
objective and pulled the shared exponent toward the four-bit optimum, even
when that choice was many times worse than the best eight-bit exponent.

This failure was particularly visible on the language model. Before correcting
the objective, the eight-bit prefix produced WikiText-2 perplexity 46.60
against a 29.00 dequantized-Q4 reference in the initial setup.

### 3.2 Normalized regret

We instead normalize every prefix by its own best attainable error:


```math
E_b^{\min} = \min_e E_b(e).
```


The relative regret of exponent $e$ at width $b$ is


```math
R_b(e) = \frac{E_b(e)} {\max(E_b^{\min},\epsilon)}.
```


The shared exponent is selected by


```math
e_c^* = \arg\min_e \sum_{b\in\mathcal B}R_b(e).
```


This objective gives each prefix comparable relative influence. A regret of
one is optimal for that width; a regret of two means twice its independently
best MSE. It does not guarantee equal downstream accuracy, but it avoids
allowing the absolute scale of four-bit error to erase the eight-bit
objective.

After this correction and the later move to the controlled BF16-source lineage,
the per-channel eight-bit Qwen prefix reached perplexity 28.16 against the BF16
source's 27.59, and group-32 eight-bit reached 27.76. The result demonstrates
that progressive scale selection is a multi-objective problem, not merely
ordinary quantization repeated at several widths.

## 4. Model-specific preparation

### 4.1 Convolutional networks

BatchNorm must be folded into the preceding convolution before quantization.
At inference, a convolution followed by BatchNorm is one affine operator.
Quantizing the unfused convolution while retaining the original BatchNorm
causes BatchNorm gains to amplify channel reconstruction error.

For ResNet-18, we therefore fused every Conv–BatchNorm pair before encoding.
The initial convolution and final classifier remained FP16 because endpoint
layers are comparatively sensitive and constitute a small part of the model.
All residual-block convolution weights were encoded dyadically.

### 4.2 Transformer language models

The language-model experiment encoded transformer linear weights while leaving
the output language-model head at higher precision. This avoids applying a
coarse code directly to the vocabulary projection, where small logit changes
affect every token probability.

LLMs expose errors that may be hidden by classification accuracy. A vision
classifier only needs the correct class to remain the largest logit. Language
model perplexity evaluates the probability assigned to every target token:


```math
PPL = \exp \left( -\frac{1}{T} \sum_{t=1}^T \log p(x_t\mid x_{< t}) \right)
```


Small perturbations therefore accumulate across layers and thousands of token
positions. Next-token agreement and perplexity are both reported because a
model can preserve many argmax predictions while still assigning materially
different probabilities.

## 5. Experimental setup

All large-model inference was executed on an Apple M5 GPU through MPS. The
scripts explicitly reject CPU fallback.

### 5.1 ResNet-18

- Model: official torchvision ResNet-18 checkpoint.
- Dataset: all 3,925 Imagenette-160 validation images.
- Encoded weights: 11,157,504 residual-block convolution weights.
- Prefix widths: 4, 5, 6, and 8 bits.
- High-precision exceptions: `conv1`, final `fc`, and non-weight state.
- Metrics: top-1 accuracy, agreement with FP16 predictions, logit MAE,
  estimated stored size, conversion time, and refinement-plane size.

### 5.2 Qwen2.5 0.5B

- Model: Qwen2.5-0.5B-Instruct, 494.03M parameters.
- Source consistency: the audited BF16 Safetensors checkpoint was evaluated
  directly in Transformers/MPS, converted once to BF16 GGUF, and then
  independently quantized to Q4_K_M, Q6_K, and Q8_0 controls.
- Encoded weights: tied embedding/output and transformer weights, with tied
  parameters encoded once.
- Prefix widths: 4, 5, 6, and 8 bits.
- Scale granularities: per-channel and group sizes 64, 32, and 16.
- WikiText-2: 8,191 evaluated next-token targets.
- ARC-Easy: 100 questions scored by conditional likelihood.
- Textual comparison: 20 ARC-Easy free-generation prompts and 10 WikiText
  continuations judged against the BF16 source.

For this model, nominal GGUF labels do not equal effective bits per weight:
Q4_K_M is 6.35 bpw, Q6_K is 8.09 bpw, and Q8_0 is 8.50 bpw. The dyadic tables
therefore report effective bits per weight including exponent metadata and
other stored model state.

The current implementation decodes prefixes to FP16 tensors before MPS
execution. Reported latency and token throughput therefore measure FP16
execution of the reconstructed weights, not packed dyadic arithmetic.

For ResNet-18, the official fused FP32 checkpoint was the encoding source.
The reference was an independent copy cast to FP16, while each dyadic
candidate was decoded from the FP32 source and then cast to FP16. Thus the
accuracy comparison measures the added effect of dyadic weight approximation
relative to FP16 inference of the same official model. It is not a comparison
against independently optimized conventional INT4, INT6, or INT8 models.

## 6. Results

### 6.1 ResNet-18

| Representation | Estimated size | Top-1 | FP16 agreement | Logit MAE |
|---|---:|---:|---:|---:|
| FP16 reference | 23.37 MB | 66.85% | 100.00% | 0.000 |
| 4-bit dyadic | 6.64 MB | 32.15% | 38.32% | 1.540 |
| 5-bit dyadic | 8.03 MB | 65.40% | 82.17% | 0.599 |
| 6-bit dyadic | 9.43 MB | 65.86% | 92.56% | 0.257 |
| 8-bit dyadic | 12.22 MB | **66.78%** | 96.20% | 0.134 |

The six-bit prefix is a useful memory-accuracy compromise. It differs from
FP16 by 0.99 percentage points while reducing estimated stored size by 59.7%.
The eight-bit prefix differs by only 0.08 points and improves agreement and
logit fidelity substantially.

One additional plane costs 1,394,688 bytes. Encoding the complete hierarchy
took 461 ms, and reconstructing one prefix took 6–9 ms.

Synchronized batch-64 MPS latency was approximately 33–34 ms at every width.
Because every prefix was materialized as FP16, this near-constant latency is
expected and is not evidence against a future packed kernel.

The CSV also contains whole-dataset `images_per_s`, but those values are not
used as comparative performance evidence. The reference was evaluated first
and absorbed data-loader startup and MPS graph-compilation costs. The
separately warmed latency loop is the appropriate evidence for the current
FP16-materialized execution path.

These values were regenerated after a reproducibility audit. An earlier
ResNet table used the preceding raw-MSE exponent selector while the surrounding
text described normalized regret. With the old selector, six-bit top-1 had
been 66.78%; with the current normalized-regret selector it is 65.86%, while
eight-bit top-1 is 66.78%. The table in this paper corresponds to the current
code. Fusion equivalence, class-index mapping, exact prefix nesting, exponent
range, and two full-run metric repetitions were checked independently.
In FP32, fusion changed checked logits only at about the $10^{-5}$ level.
On the full FP16/MPS validation set, fused and unfused graphs had the same
66.8535% top-1 accuracy, while agreeing on 99.9236% of predictions with
0.00478 logit MAE. Thus the reported reference is specifically the fused FP16
graph; it is accuracy-equivalent here, but not bit-identical to the unfused
FP16 execution.

### 6.2 Qwen2.5 0.5B

The language-model comparison uses an audited BF16 Safetensors source, one BF16
GGUF converted from that source, and independently quantized GGUF controls
(Q4_K_M, Q6_K, Q8_0) produced from the BF16 GGUF. The GGUF names are nominal:
for this 896-wide model, Q4_K_M stores many tensors at Q5_0, Q6_K, or Q8_0 and
the large embedding at high precision. Effective bits per weight are therefore
the fair comparison unit.

The original per-channel dyadic variant uses one exponent per output row. The
block-wise variant keeps the same sign and magnitude planes, but assigns one
power-of-two exponent to each contiguous group of weights. The following sweep
uses WikiText-2 (8,191 target tokens) and ARC-Easy conditional likelihood (100
questions), decoded from the BF16 source on MPS:

| Group | bits | effective bpw | WikiText-2 PPL | next-token agreement | ARC-Easy |
|---|---:|---:|---:|---:|---:|
| per-channel | 4 | 4.01 | 102.76 | 44.4% | 30% |
| 64 | 4 | 4.13 | 53.13 | 56.3% | 38% |
| 32 | 4 | 4.25 | 46.40 | 60.0% | 33% |
| 16 | 4 | 4.50 | 39.81 | 65.0% | 39% |
| per-channel | 5 | 5.01 | 34.87 | 71.3% | 36% |
| 32 | 5 | 5.25 | 31.84 | 76.1% | 47% |
| per-channel | 6 | 6.01 | 29.65 | 83.7% | 43% |
| 64 | 6 | 6.13 | 28.61 | 87.4% | 46% |
| 32 | 6 | 6.25 | 28.88 | 87.0% | 47% |
| per-channel | 8 | 8.01 | 28.16 | 91.5% | 43% |
| 32 | 8 | 8.25 | 27.76 | 94.5% | 43% |

Block-wise exponents improve every width and especially the four-bit prefix:
perplexity falls from 102.76 to 53.13 with group 64 and to 39.81 with group 16.
Group 32 is the main operating point in the rest of this section because it
captures most of the quality gain while adding only 0.25 bits/weight of exponent
metadata.

Against the independently built controls:

| Representation | effective bpw | estimated size | WikiText-2 PPL | next-token agreement | ARC-Easy |
|---|---:|---:|---:|---:|---:|
| BF16 source | 16.00 | 988.10 MB | 27.59 | 100.0% | 40% |
| dyadic g32, 6-bit | 6.25 | 386.05 MB | 28.88 | 87.0% | 47% |
| Q4_K_M | 6.35 | 397.81 MB | 28.64 | 87.9% | 37% |
| dyadic g32, 8-bit | 8.25 | 509.54 MB | 27.76 | 94.5% | 43% |
| Q6_K | 8.09 | 505.74 MB | 27.86 | 94.5% | 39% |
| Q8_0 | 8.50 | 531.07 MB | 27.53 | 96.0% | 40% |

At roughly 6.3 bpw, group-32 dyadic matches Q4_K_M on perplexity and
next-token agreement while using a single progressively refinable code. At
roughly 8.2 bpw, it matches Q6_K and approaches Q8_0. This comparison does not
make the dyadic code a drop-in replacement for GGUF: the current prototype
still materializes FP16 tensors for execution, while GGUF has mature packed
kernels. It does show that the quality cost of progressivity is small once the
scale granularity is block-wise rather than per-channel.

The separate native Ollama BF16-source run reported:

| Metric | Result |
|---|---:|
| Exact-letter ARC-Easy accuracy | 63% |
| Valid answer format | 98% |
| Prompt throughput | 4,456 tokens/s |
| Generation throughput | 219 tokens/s |
| Processor | 100% GPU |

This ARC result uses free generation and a chat prompt, whereas the table above
uses conditional likelihood in Transformers/MPS. They are different protocols and
must not be compared as the same statistic.

### 6.3 Textual-output equivalence

The agreement and likelihood metrics above measure how a variant *scores*
tokens. They do not measure whether the text it *generates* still means the same
thing: greedy decoding turns any logit difference into a divergent token stream,
so two outputs can differ from the first token yet reach the same conclusion. To
test meaning rather than surface form, the BF16-source controlled lineage and
its dyadic prefixes and GGUF controls (Q4_K_M, Q6_K, Q8_0) were each made to
generate greedily from identical prompts in one backend, and every variant's
text was compared to the BF16 source.

Surface agreement is uninformative here: no variant reproduced the source text
verbatim on any prompt, including Q8_0. The meaningful signals are an embedding
cosine (a local `nomic-embed-text` model) and a same-meaning verdict from an
independent LLM judge of a different model family.

On 20 ARC-Easy instruction prompts (full free-form answers, not the scored
letter), the group-32 dyadic prefixes compare as follows:

| Variant | effective bpw | mean cosine | meaning-equivalent |
|---|---:|---:|---:|
| dyadic g32, 4-bit | 4.25 | 0.875 | 30% |
| dyadic g32, 5-bit | 5.25 | 0.921 | 90% |
| dyadic g32, 6-bit | 6.25 | 0.959 | 100% |
| dyadic g32, 8-bit | 8.25 | 0.974 | 95% |
| Q4_K_M | 6.35 | 0.900 | 55% |
| Q6_K | 8.09 | 0.946 | 95% |
| Q8_0 | 8.50 | 0.942 | 90% |

The group-32 change materially alters the textual conclusion. Four bits remain
weak, but no longer collapse completely. Five bits preserve the source answer
on 90% of the instruction prompts. Six-bit group-32 dyadic reaches full
meaning-equivalence on this sample at 6.25 bpw, exceeding Q4_K_M at 6.35 bpw
and matching or exceeding the higher-bit GGUF controls under the same judge.
Eight-bit group-32 dyadic has the highest deterministic embedding cosine of
any tested variant.

On 10 open-ended WikiText continuations, meaning-equivalence is lower for every
variant because unconstrained greedy continuation diverges quickly regardless
of quantization. The group-32 eight-bit prefix is still the strongest variant
by both metrics (cosine 0.868 and 50% judge equivalence, versus Q8_0 at 0.702
and 20%). The verbatim metric alone would have understated the dyadic prefixes:
exact matches are nearly absent, while semantic metrics show that block-wise
5-8-bit dyadic text carries the same instruction-following meaning as both the
source and the GGUF controls.

## 7. Discussion

### 7.1 What the experiments establish

First, one maximum-depth dyadic code can provide several operationally useful
prefixes. On ResNet-18, six bits came within one percentage point of FP16 and
eight bits came within 0.08 points. On Qwen, group-32 six-bit dyadic matched
the main probability metrics of Q4_K_M at comparable effective size and
preserved the BF16 source's ARC-Easy instruction meaning on the textual sample.

Second, progressive refinement has a concrete deployment meaning. A device can
load a lower-width prefix and later receive exactly one additional bit per
weight. Earlier planes, scale metadata, and the model structure remain
unchanged. This is useful for memory-adaptive loading, staged downloads, model
streaming, and heterogeneous devices.

Third, the representation requires model-aware preparation. Controlled clipping,
normalized regret, and scale granularity are not optional refinements; naive
range-covering power-of-two scales and overly coarse per-channel exponents
caused severe accuracy collapse. Likewise, BatchNorm fusion was required for
the convolutional model.

Fourth, selecting a scale for multiple prefixes is genuinely different from
ordinary single-width quantization. Raw summed MSE implicitly prioritized the
lowest precision. Normalized regret produced a substantially better compromise
and recovered near-reference eight-bit language-model perplexity.

### 7.2 What remains unproven

The experiments do not demonstrate a speedup over optimized INT4, INT8, or
GGUF kernels. The stored representation is low-bit, but the current execution
path reconstructs FP16 tensors. Power-of-two scales make a packed implementation
plausible; they do not make it automatic.

Likewise, the reported model sizes are estimates for a proposed packed format:
one packed sign/magnitude payload, one signed-byte exponent per channel or
group, and FP16 storage for excluded parameters and buffers. They are not sizes
of serialized dyadic model files produced by the current prototype.

The language-model comparison is also not a direct systems contest with native
Q4_K_M storage. Q4_K_M uses block quantization, specialized metadata, and
optimized Metal kernels. The dyadic prototype is only a quality and storage
model: it stores the nested code in Python objects and then materializes FP16
tensors for execution. A fair systems comparison requires a native dyadic file
format and kernel.

The exponent objective remains weight-local. It does not account for input
activation covariance, Hessian sensitivity, or output-logit effects.
Activation-aware or loss-aware calibration may improve the five- and six-bit
prefixes substantially.

Finally, only one convolutional model and one small language model were tested.
The observed six-bit operating point should be treated as evidence, not a
universal constant.

## 8. Systems implications and future work

A useful packed implementation should preserve the progressive property rather
than first combining all planes into a conventional integer tensor. A Metal or
llama.cpp kernel could:

1. read sign and magnitude planes in MSB-first order;
2. stop after the number of planes permitted by the memory or latency budget;
3. accumulate each plane with its corresponding binary shift;
4. apply the channel or group exponent through fixed-point scaling;
5. accept an additional plane without rewriting earlier model data.

There are two likely execution strategies. A bit-serial kernel would process
one plane at a time and could stop early, maximizing adaptivity. A fused
materialization kernel would combine selected planes into INT4 or INT8 blocks
once, then use existing optimized GEMM machinery. The latter sacrifices
plane-by-plane compute adaptivity but may be easier to integrate and could
retain the incremental storage and transfer benefits.

Further algorithmic work should consider:

- broader sweeps of group size and tensor-specific group-size schedules;
- activation-weighted normalized regret;
- calibration by output-logit KL divergence or language-model loss;
- joint fine-tuning over all requested prefixes;
- nonuniform allocation of planes across layers;
- keeping sensitive attention, embedding, or output tensors at higher width
  through mixed prefix widths;
- comparisons against Q4_K_M, Q6_K, Q8_0, GPTQ, AWQ, and standard INT4/INT8
  using identical source weights and evaluation sets.

## 9. Conclusion

Progressive dyadic quantization is a viable representation for
memory-adaptive neural deployment. Its central advantage is not a new notion of
real-number proximity or native two-adic neural arithmetic. It is the ability
to encode a model once and expose increasingly precise, executable prefixes
whose storage differs by exactly one bit per weight.

The large-model experiments identify the conditions under which this idea
works in practice: power-of-two exponents at appropriate granularity,
controlled clipping, midpoint interval decoding, model-specific operator
fusion, and normalized-regret optimization across all intended prefix widths.
Under these conditions, six-bit dyadic weights gave near-reference ResNet-18
accuracy and group-32 six-bit Qwen weights matched the probability-quality
band of Q4_K_M while preserving ARC-Easy instruction meaning in free
generation; eight-bit weights preserved ResNet-18 accuracy within 0.08
percentage points and matched the higher-bit Qwen controls closely.

The evidence is therefore positive but incomplete. Numerical quality,
progressive refinement, conversion cost, and theoretical storage have been
demonstrated. The decisive remaining question is whether a packed
implementation can convert the representation's simple shifts, masks, and
incremental planes into a latency or energy advantage over mature low-bit
kernels.

## References

1. C. Weiß, “P-adic Poissonian Pair Correlations via the Monna Map,”
   <https://arxiv.org/abs/2406.13255>.
2. R. Lubarsky and F. Richman, “Signed-bit representations of real numbers,”
   <https://arxiv.org/abs/1510.00648>.
3. F. Wiesnet and N. Köpp, “Limits of real numbers in the binary signed digit
   representation,” <https://arxiv.org/abs/2103.15702>.
4. H. Li, J. J. Davis, J. Wickerson, and G. A. Constantinides, “ARCHITECT:
   Arbitrary-precision Hardware with Digit Elision for Efficient Iterative
   Compute,”
   <https://arxiv.org/abs/1910.00271>.
5. K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image
   Recognition,” <https://arxiv.org/abs/1512.03385>.
6. Qwen Team, “Qwen2.5 technical report,”
   <https://arxiv.org/abs/2412.15115>.

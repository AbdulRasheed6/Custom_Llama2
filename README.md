# Llama 2 Introduction

The Llama 2 architecture took its inspiration from the **transformer architecture** but differs from other instances of second-generation transformers like GPT-2, GPT-3, and DeepSeek.

---

## Structural Components of the Llama 2 Block

1. **Token Embeddings**  
   The input data are tokenized with the byte pair encoding algorithm for computational efficiency.

2. **Pre-Normalisation**  
   The input of each transformer sub-layer is normalized using Root Mean Square Normalisation (RMSNorm) to improve training stability. The sub-layers of the transformer consist of:  
   - **Input layer**  
   - **Self-attention layer (GQA + RoPE)**  
   - **Feed-forward layer**  
   - **Output layer**

3. **Grouped Query Masked Self-Attention (GQA + MSA)**  
   This is the communication layer of the **Llama 3** architecture, **although Multi head attention was used for Llama2**, but I decided to use GQA since it is the only major architectural difference from Llama2 . The GQA differs slightly from the GPT-2 self-attention mechanism because it  share key/value heads across query heads.  
   - Queries have more heads than keys/values.  
   - Keys/values are reused across queries, lowering redundancy.  

   **Benefits:**  
   - Faster inference  
   - Lower memory footprint  
   - Comparable performance to full multi-head attention  

   Combined with Masked Self-Attention (MSA), GQA ensures tokens cannot attend to future positions, preserving autoregressive training.

4. **Rotary Embeddings (RoPE)**  
   Applied directly to the query and key vectors to provide rotary positional embeddings. Unlike absolute positional embeddings, RoPE considers the relative position of tokens.

5. **Feed Forward Network (FFN)**  
   If attention is communication, the FFN is "thinking." This is a linear layer followed by a non-linearity (SWiGLU). It acts as a **knowledge bank**, processing the information gathered during the attention phase.

6. **Residual Network**  
   The input of the block is added back to its output. This allows gradients to flow through the network during backpropagation without vanishing, enabling deeper models.

---

## Data Dimensions & Tensors

We handle data in 4D or 3D tensors. For our implementation, the core shape is:

- **B (Batch):** 64 (Processing 64 sequences at once)  
- **T (Time/Block Size):** 4096 (The maximum context length/history the model sees)  
- **C (Channel/Embed Dim):** `n_embed` (The size of the vector representing a single token)  
- **n_heads (Number of heads):** Determines how many independent attention patterns the model can learn simultaneously  
- **n_kv_heads:** 8 (Controls the memory cost of generation)

---

## Mechanics of Masked Self-Attention

Self-attention allows tokens to "vote" on which other tokens are relevant to them.

### Mathematical Flow

We derive three vectors for every token using trainable weights:

1. **Query (Q):** What am I looking for?  
2. **Key (K):** What information do I contain?  
3. **Value (V):** If I am relevant, what information do I contribute?
4. **head_size  \(d_k\):** Dimension of the key vectors (also called **head size**)

**Score Calculation:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$


Steps:
- **Matrix Multiplication:** Creates an affinity matrix showing how much every token relates to others.  
- **Scaling:** $\sqrt{d_k}$ to stabilize gradients.  
- **Mask (Tril):** Applies a lower-triangular mask (`torch.tril`) to block future tokens.  
- **Softmax:** Normalizes values to sum to 1, creating a weighted map of the past.  
- **Value Aggregation:** Multiplies this map by \(V\) to get the final context-aware representation.

---

## Mechanics of Root Mean Square Normalisation (RMSNorm)

RMSNorm ensures data stability. It is different from LayerNorm as it is computationally efficient and removes mean subtraction.


$$
\bar{a}_i = \frac{a_i}{\text{RMS}(\mathbf{a})} \, g_i, 
\quad \text{where} \quad 
\text{RMS}(\mathbf{a}) = \sqrt{\frac{1}{n} \sum_{i=1}^n a_i^2 + \epsilon}
$$

Where :

- $\bar{a}_i$: Represents the normalised output
-  $\text{RMS}(\mathbf{a})$: The root mean square of the input vector
- $g_i$: The learnable  gain parameter (scaling factor).
- $\epsilon$: To avoid zero division error

---

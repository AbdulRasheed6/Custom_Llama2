from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# 2. Download Tiny Shakespeare Dataset (Andrej Karpathy)
# ============================================================
# This works directly in Google Colab
!wget -q https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# ============================================================
# 3. Tokenizer (LLaMA tokenizer, NOT GPT-2)
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    use_fast=False
)

tokenizer.pad_token = tokenizer.eos_token

# Encode entire dataset
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

# Train / validation split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# ============================================================
# 4. Training hyperparameters
# ============================================================
batch_size = 8
seq_len = 256
lr = 3e-4
max_steps = 2000

# ============================================================
# 5. Data loader
# ============================================================
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    
    x = torch.stack([data[i:i+seq_len] for i in ix])          # (B, T)
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])      # (B, T)
    
    return x.to(device), y.to(device)



# ===============================
# Config
# ===============================
@dataclass
class LlamaConfig:
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    seq_len: int
    base: float = 10000.0
    eps: float = 1e-5
    multiple_of: int = 256   # LLaMA uses power-of-two alignment


# ===============================
# RMSNorm
# ===============================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


# ===============================
# Rotary Embedding (RoPE)
# ===============================
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, base):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.cos = None
        self.sin = None
        self.seq_len_cached = 0

    def _build_cache(self, seq_len, device, dtype):
        if self.seq_len_cached >= seq_len:
            return
        self.seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos = emb.cos()[None, None, :, :].to(dtype)
        self.sin = emb.sin()[None, None, :, :].to(dtype)

    def rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k):
        seq_len = q.size(-2)
        self._build_cache(seq_len, q.device, q.dtype)
        cos = self.cos[:, :, :seq_len]
        sin = self.sin[:, :, :seq_len]
        q = q * cos + self.rotate_half(q) * sin
        k = k * cos + self.rotate_half(k) * sin
        return q, k


# ===============================
# SwiGLU MLP (derived hidden dim)
# ===============================
class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(8 * config.dim / 3)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        self.gate = nn.Linear(config.dim, hidden_dim, bias=False)
        self.up = nn.Linear(config.dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, config.dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ===============================
# Grouped Query Attention
# ===============================
class LlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.scale = self.head_dim ** -0.5
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(config.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)

        self.rotary = RotaryEmbedding(self.head_dim, config.base)

    def forward(self, x):
        B, T, C = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = self.rotary(q, k)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device),
            diagonal=1
        )
        attn = attn + mask[None, None, :, :]

        attn = F.softmax(attn, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.wo(out)


# ===============================
# Transformer Block
# ===============================
class LlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, config.eps)
        self.attn = LlamaAttention(config)
        self.ffn_norm = RMSNorm(config.dim, config.eps)
        self.mlp = LlamaMLP(config)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x


# ===============================
# LLaMA Model
# ===============================
class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([LlamaBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, config.eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying
        self.config = config

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


# ===============================
# Training + Generation Demo
# ===============================
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token

    text = "To be or not to be, that is the question."
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    config = LlamaConfig(
        dim=512,
        n_layers=6,
        n_heads=8,
        n_kv_heads=4,
        vocab_size=tokenizer.vocab_size,
        seq_len=128
    )

    model = LlamaModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    def get_batch(batch_size=4):
        ix = torch.randint(len(data) - config.seq_len - 1, (batch_size,))
        x = torch.stack([data[i:i+config.seq_len] for i in ix])
        y = torch.stack([data[i+1:i+config.seq_len+1] for i in ix])
        return x, y

    model.train()
    for step in range(200):
        xb, yb = get_batch()
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), yb.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"step {step} | loss {loss.item():.4f}")

    @torch.no_grad()
    def generate(prompt, max_new_tokens=50):
        model.eval()
        input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
        for _ in range(max_new_tokens):
            input_ids = input_ids[:, -config.seq_len:]
            logits = model(input_ids)
            next_token = torch.argmax(logits[:, -1], dim=-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
        return tokenizer.decode(input_ids[0])

    print("\nGenerated:")
    print(generate("First Citizen:"))


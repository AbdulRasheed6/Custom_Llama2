# =========================================================
# LLaMA from Scratch + Tiny Shakespeare (Colab Ready)
# nanoGPT-style character encoding
# =========================================================

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import urllib.request

# =========================================================
# Device
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# Download Tiny Shakespeare (Colab-safe)
# =========================================================
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "/content/tiny_shakespeare.txt"

if not os.path.exists(DATA_PATH):
    print("Downloading Tiny Shakespeare dataset...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

print(f"Dataset size: {len(text):,} characters")

# =========================================================
# nanoGPT-style Character Encoding
# =========================================================
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)   # (N,)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print("Vocab size:", vocab_size)

# =========================================================
# Config
# =========================================================
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
    multiple_of: int = 128


# =========================================================
# RMSNorm
# =========================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))   # (D,)
        self.eps = eps

    def forward(self, x):
        # x: (B, T, D)
        return self.weight * x * torch.rsqrt(
            x.pow(2).mean(-1, keepdim=True) + self.eps
        )


# =========================================================
# Rotary Embedding
# =========================================================
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, base):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.cos = None
        self.sin = None
        self.seq_len_cached = 0

    def _build_cache(self, seq_len, device, dtype):
        if seq_len <= self.seq_len_cached:
            return
        self.seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device).float()              # (T,)
        freqs = torch.outer(t, self.inv_freq)                         # (T, Hd/2)
        emb = torch.cat((freqs, freqs), dim=-1)                       # (T, Hd)
        self.cos = emb.cos()[None, None, :, :].to(dtype)              # (1,1,T,Hd)
        self.sin = emb.sin()[None, None, :, :].to(dtype)              # (1,1,T,Hd)

    def rotate_half(self, x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k):
        # q, k: (B, H, T, Hd)
        seq_len = q.size(-2)
        self._build_cache(seq_len, q.device, q.dtype)
        cos = self.cos[:, :, :seq_len]
        sin = self.sin[:, :, :seq_len]
        return (
            q * cos + self.rotate_half(q) * sin,
            k * cos + self.rotate_half(k) * sin
        )


# =========================================================
# SwiGLU MLP (NO hardcoded hidden dim)
# =========================================================
class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(8 * config.dim / 3)
        hidden_dim = config.multiple_of * (
            (hidden_dim + config.multiple_of - 1) // config.multiple_of
        )

        self.gate = nn.Linear(config.dim, hidden_dim, bias=False)
        self.up = nn.Linear(config.dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, config.dim, bias=False)

    def forward(self, x):
        # x: (B, T, D)
        return self.down(F.silu(self.gate(x)) * self.up(x))


# =========================================================
# Grouped Query Attention
# =========================================================
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
        # x: (B, T, D)
        B, T, D = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = self.rotary(q, k)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        attn = (q @ k.transpose(-2, -1)) * self.scale      # (B, H, T, T)

        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )
        attn = attn + causal_mask[None, None, :, :]

        attn = F.softmax(attn, dim=-1)
        out = attn @ v                                    # (B, H, T, Hd)
        out = out.transpose(1, 2).reshape(B, T, D)        # (B, T, D)
        return self.wo(out)


# =========================================================
# Transformer Block
# =========================================================
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


# =========================================================
# LLaMA Model
# =========================================================
class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([LlamaBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, config.eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.config = config

    def forward(self, input_ids):
        # input_ids: (B, T)
        x = self.embed(input_ids)                          # (B, T, D)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))                  # (B, T, V)


# =========================================================
# Training Setup
# =========================================================
config = LlamaConfig(
    dim=384,
    n_layers=6,
    n_heads=6,
    n_kv_heads=3,
    vocab_size=vocab_size,
    seq_len=128
)

model = LlamaModel(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

def get_batch(batch_size=16):
    data = train_data
    ix = torch.randint(len(data) - config.seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i+config.seq_len] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+config.seq_len+1] for i in ix]).to(device)
    return x, y


# =========================================================
# Training Loop
# =========================================================
model.train()
for step in range(2000):
    xb, yb = get_batch()
    logits = model(xb)
    loss = F.cross_entropy(logits.view(-1, config.vocab_size), yb.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(f"Step {step} | Loss {loss.item():.4f}")


# =========================================================
# Text Generation
# =========================================================
@torch.no_grad()
def generate(prompt, max_new_tokens=300):
    model.eval()
    input_ids = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        input_ids = input_ids[:, -config.seq_len:]
        logits = model(input_ids)
        next_token = torch.argmax(logits[:, -1], dim=-1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)

    return decode(input_ids[0].tolist())


print("\n================= GENERATED SHAKESPEARE =================\n")
print(generate("ROMEO:"))

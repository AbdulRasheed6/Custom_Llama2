from dataclasses  import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from transformers import AutoTokenizer
from typing import Optional, Tuple



class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # precompute inverse frequencies
        # theta_i= base^(-2i/dim) for i=0,2,4......, dim-2
        inv_freq= 1.0/(config.base**(torch.arange(0, config.dim, 2, dtype=torch.float32) /config.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached =0
        
        self.cos_cached=None
        self.sin_cached=None
        
        
    def _rotate_half(self, x):
        """Rotate half of the hidden dimensoins. """
        # x :(B, n_heads, T, head_dim)
        x1= x[..., : x.shape[-1]//2] # first half
        x2= x[..., x.shape[-1]//2:] # second half
        return torch.cat((-x2, x1), dim=-1) # rotated
    

    def _apply_rotary_emb(self, q:torch.Tensor, k:torch.Tensor, cos:torch.Tensor, sin:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        #q, k: (B, n_heads, T, head_dim)
        # cos, sin: (1,1, T, head_dim)

        q_embed= (q*cos) + (self._rotate_half(q)*sin) #(B, n_heads,  T, head_dim)
        k_embed= (k*cos) + (self._rotate_half(k)* sin) #(B, n_heads, T, head_dim)

        return q_embed, k_embed

        


    def _build_cos_sin_cache(self, seq_len, device: torch.device, dtype:torch.dtype):
        if seq_len <= self.max_seq_len_cached and self.cos_cached is not None:
            return
        
        self.max_seq_len_cached=seq_len

        m= torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs= torch.outer(m, self.inv_freq) # (Mn* ThETAn) -> (T, head_dim//2)

        # duplicate to full dim
        emb= torch.cat((freqs, freqs), dim=-1) #(sqe_len, head_dim) 

        cos= emb.cos()[None, None, :, :].to(dtype) # (1,1 seq_len, head_dim)
        sin= emb.sin()[None, None, :, :].to(dtype)

    
        self.register_buffer("cos_cached", cos, persistent=False) 
        self.register_buffer("sin_cached", sin, persistent=False)


        
    def forward(self, x, position_ids: Optional[torch.Tensor]= None) -> Tuple[torch.Tensor, torch.Tensor]:


        # x: (B, num_heads, T, head_dim)
        seq_len= x.shape[-2] if position_ids is None else position_ids.shape[-1]
        

        if seq_len> self.max_seq_len_cached or self.cache is None:
            self._build_cos_sin_cache(seq_len, x.device, x.dtype)


        # slice current positions
        cos= self.cos_cached[:, :, :seq_len, :]  #(1, 1, T, head_dim)
        sin = self.sin_cached[:, :, :seq_len, :]  #(1, 1, T, head_dim)
    

        if position_ids is not None:
            # position_ids: (B, T)
            cos= cos.squeeze(0).squeeze(0)  #(T,head_dim)
            sin= sin.squeeze(0).squeeze(0) # (T, head_dim)
            cos= cos[position_ids] #(B, T, head_dim)
            sin= sin[position_ids]
            cos= cos.unsqueeze(1)
            sin= sin.unsqueeze(1)
        
        return cos, sin #(B, n_heads, T, head_dim) 

class RMSNorm(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config= config
        self.num_eps= self.config.num_eps
        self.weight= nn.Parameter(torch.ones(config.dim))

    def forward(self, x):
        #x: (B, T, n_embed)
        input_dtype= x.dtype
        x= x.to(torch.float32)

        # compute the variance: mean(x**2, dim=-1)
        
        variance= x.pow(2).mean(-1, keepdim=True)

        x= x * torch.rsqrt(variance + self.num_eps)
        out= x *  self.weight
        return out.to(input_dtype)


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.gate_proj= nn.Linear(config.dim, config.hidden_dims, bias= False)  #(4096 -> 11008)
        self.up_proj= nn.Linear(config.dim, config.hidden_dims, bias=False) #(4096 -> 11008)
        self.down_proj= nn.Linear(config.hidden_dims, config.dim,  bias=False)  #(11008, 4096)



    def forward(self, x):
        gate= F.silu(self.gate_proj(x))  #(B, T, hidden_dims)
        up= self.up_proj(x)    #(B, T, hidden_dims)
        return self.down_proj(gate * up)  #(B, T, n_emebed)
        
    


class LlamaGQA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads= config.n_heads
        self.n_kv_heads= config.n_kv_heads

        self.head_dim= config.head_dim
        self.n_rep= self.n_heads// self.n_kv_heads
        self.scale= self.head_dim ** -0.5



        self.wq= nn.Linear(config.dim, config.n_heads* self.head_dim, bias=False) 
        self.wk= nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv= nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo= nn.Linear(config.dim, config.n_embed, bias=False)

        self.rotary_emb= LlamaRotaryEmbedding(config)

    def forward(self, x, position_ids, mask: Optional[torch.Tensor]=None):

        B, T, C= x.shape #(B, T, C=m_embed)
        q= self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k= self.wk(x).view(B, T, self.n_heads, self.head_dim)
        v= self.wv(x).view(B, T, self.n_heads, self.head_dim)

        cos, sin= self.rotary_emb(q, position_ids=position_ids)
        q, k= self.rotary_emb._apply_rotary_emb(q, k , cos , sin) # (B, T,n_heads, head_dim)

        q= q.transpose(1,2) # (B, n_heads, T, head_dim)
        k= k.transpose(1,2)
        v= v.transpose(1,2)

        # GQA: Repeat KV eads to matc Query eads

        if self.n_rep>1:
            k= k.repeat_interleave(self.n_rep, dim=1)
            v= v.repeat_interleave(self.n_rep, dim=1)


        scores=torch.matmul(q,k.transpose(-2, -1)) * self.scale #(B, n_heads, T,T)
        if mask is not None:
            scores= scores + mask

        attn= F.softmax(scores.float(), dim=-1).type_as(q) #(B, n_heads, T, T)
        output= torch.matmul(attn, v)

        output= output.transpose(1,2).contiguous().view(B, T, C)
        return self.wo(output)


class LlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm= RMSNorm(config)
        self.attn= LlamaGQA(config)
        self.ffn_norm= RMSNorm(config)
        self.ffn= LlamaMLP(config)


    def forward(self, x, position_ids, mask: Optional[torch.Tensor]):

        # x : (B, T, C) C=n_embed
        g= x + self.attn(self.attn_norm(x), position_ids, mask)
        out= g + self.ffn(self.ffn_norm(g))
        return out



@dataclass
class LlamaConfig:
    dim: int= 4096
    n_layers: int= 32
    n_heads: int= 32 #Number of eads for te queries
    n_kv_heads: int= 8 #Number of heads for K and V
    vocab_size: int= 32000 # it is set wen te tokenizer is loaded
    num_eps: float= 1e-5
    hidden_dims= 11008
    head_dim= 4096// 32  #dim // n_heads
    base: float= 10000.0

    seq_len : int=2048







class Llama2(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size !=-1, "Vocab size must be set"
        
        self.config= config
        self.n_layers= config.n_layers
        self.vocab_size= config.vocab_size


        self.tok_embeddings= nn.Embedding(self.vocab_size, config.dim)
        self.layers= nn.ModuleList([LlamaBlock(config) for _ in range(self.n_layers)])

        self.norm= RMSNorm(config)

        
        lm_head= nn.linear(config.dim, self.vocab_size)

        #Weight typing(saving parameters)
        self.lm_head.weight= self.tok_embeddings.weight
    

    def forward(self, input_ids, position_ids: Optional[torch.Tensor]= None):

        # input_ids: (B, T)

        B, T= input_ids.shape()
        x= self.tok_embeddings(input_ids) # (B,T, n_embed)

        if position_ids is None:
            position_ids= torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
            
        mask= None
        if T>1:
            mask= torch.full((T,T), float("-inf"), device= x.device)
            mask= torch.triu(mask, diagonal=1)
            mask= mask.unsqueeze(0).unsqueeze(0) # (1,1, T, T)

        for layer in self.layers:
            x= layer(x, position_ids, mask) #(B,T, n_embed)

        x= self.norm(x)
        return self.lm_head(x)
        


# Training and  Generation Example

if __name__=="__main__":
    #use Llama tokenizer
    tokenizer= AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")

    # your text dataset
    text=""

    encoded= tokenizer.encode(text)
    data= torch.tensor(encoded, dtype=torch.long)


    batch_size=4
    block_size=128


    def get_batch():
        ix= torch.randint(len(data) - block_size, (batch_size, ))
        x= torch.stack([data[i:i+block_size] for i in ix])
        y= torch.stack([data[i+1: block_size+1] for i in ix])
        return x, y




    model= Llama2( 
        vocab_size= tokenizer.vocab_size,
        dim=512,
        n_layers=6,
        n_heads=8, 
        n_kv_heads=4, # GQA active: 8 Query heads, 4 KV heads
        hidden_dim= 2048
    )

    optimizer= torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()

    for step in range(1000):
        xb, yb= get_batch()
        position_ids= torch.arange(block_size, device=xb.device).unsqueeze(0).expand(batch_size, -1)
        logits= model(xb, position_ids=position_ids)
        loss= F.cross_entropy(logits.view(-1, tokenizer.vocab_size), yb.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 ==0:
            print(f"Step {step} | Loss : {loss.item():.4f}")


    
    @torch.no_grad()
    def generate(prompt: str, max_new_tokens: int=100):
        model.eval()
        input_ids= torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)


        for _ in range(max_new_tokens):
            seq_len= input_ids.size(1)
            position_ids=torch.arange(seq_len).unsqueeze(0)
            logits=model(input_ids, position_ids=position_ids)
            next_token= torch.argmax(logits[:,-1, :], dim=-1)
            input_ids= torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)

        return tokenizer.decode(input_ids[0])
    

    print("\n--------- Generated Sakspare")
    print(generate("First Citizen:"))


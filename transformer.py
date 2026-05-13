import torch
import numpy as np
import torch.nn as nn

def pos_enc(pos, emb_dim):
	# return the embedding vector of pos
	emb = torch.zeros(emb_dim)
	k = emb_dim // 2
	r = emb_dim % 2

	for i in range(k):
		emb[2*i] = np.sin(pos/10000**(2*i/emb_dim))
		emb[2*i+1] = np.cos(pos/10000**(2*i/emb_dim))

	if r:
		emb[-1] = np.sin(pos/10000**(2*(k+1)/emb_dim))

	return emb

def position_encoding(seq_len, emb_dim, device='cpu'):
	# return a (seq_len, emb_dim) tensor whose columns are 
	# embedding vectors of the positions 0 thru seq_len - 1
	seq_emb = []

	for pos in range(seq_len):
		seq_emb.append(pos_enc(pos, emb_dim))

	return torch.stack(seq_emb).to(device)

def causal_mask(seq_len, device='cpu'):
	# create a causal mask of size (seq_len, seq_len)
	mask = torch.zeros(seq_len, seq_len)
	indices = torch.triu_indices(seq_len, seq_len, offset=1)
	mask[indices[0], indices[1]] = 1

	return mask.bool().to(device)


class transformer_block(nn.Module):

	def __init__(self, emb_dim, num_heads, is_causal=False, device='cpu'):
		super().__init__()
		self.is_causal = is_causal
		self.device = device

		self.attention = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True, dropout=0.1).to(device)
		self.norm1 = nn.LayerNorm(emb_dim).to(device)
		
		self.mlp = nn.Sequential(
			nn.Linear(emb_dim, 4 * emb_dim),
			nn.GELU(),
			nn.Linear(4 * emb_dim, emb_dim)
			).to(device)
		self.norm2 = nn.LayerNorm(emb_dim).to(device)


	def forward(self, x):

		seq_len = x.size()[1]

		# (causally masked) attention, residual connection, layer normalization
		mask = causal_mask(seq_len, self.device) if self.is_causal else None
		attn, _ = self.attention(x, x, x, attn_mask=mask, is_causal=self.is_causal)
		x = self.norm1(x + attn)

		# MLP, residual connection, layer normalization
		ff = self.mlp(x)
		return self.norm2(x + ff)


class transformer(nn.Module):
	# uses learned positional embedding instead of fixed 

	def __init__(self, emb_dim, seq_len, num_heads, num_blocks, num_tokens, is_causal=False, device='cpu'):
		super().__init__()
		self.emb_dim = emb_dim
		self.seq_len = seq_len
		self.is_causal = is_causal
		self.device = device

		self.token_emb = nn.Embedding(num_embeddings=num_tokens, embedding_dim=emb_dim).to(device)
		self.pos_emb = nn.Embedding(num_embeddings=seq_len, embedding_dim=emb_dim).to(device)

		transformer_blocks = []
		for _ in range(num_blocks):
			transformer_blocks.append(transformer_block(emb_dim, num_heads, is_causal, device))

		self.transformer_blocks = nn.Sequential(*transformer_blocks)

		self.to_logits = nn.Linear(emb_dim, num_tokens).to(device)

	def forward(self, x):

		batch_size = x.size()[0]

		# embed the sequence x of tokens and add the position encoding^{\otimes batch_size}
		x = self.token_emb(x) + self.pos_emb(torch.arange(self.seq_len, device=self.device))[None].expand(batch_size, -1, -1)

		# feed thru the transformer blocks and convert to logits
		x = self.transformer_blocks(x)
		return self.to_logits(x)


class transformer2(nn.Module):

	def __init__(self, emb_dim, num_heads, num_blocks, num_tokens, is_causal=False, device='cpu'):
		super().__init__()
		self.emb_dim = emb_dim
		self.is_causal = is_causal
		self.device = device

		self.token_emb = nn.Embedding(num_embeddings=num_tokens, embedding_dim=emb_dim).to(device)

		transformer_blocks = []
		for _ in range(num_blocks):
			transformer_blocks.append(transformer_block(emb_dim, num_heads, is_causal, device))

		self.transformer_blocks = nn.Sequential(*transformer_blocks)

		self.to_logits = nn.Linear(emb_dim, num_tokens).to(device)

	def forward(self, x):
		# x is a (batch_size, seq_len) tensor whose entries are token ids
		batch_size, seq_len = x.size()

		# embed the sequence x of tokens and add the position encoding^{\otimes batch_size}
		x = self.token_emb(x) * (self.emb_dim ** 0.5) + position_encoding(seq_len, self.emb_dim, self.device)[None].expand(batch_size, -1, -1)

		# feed thru the transformer blocks and convert to logits
		x = self.transformer_blocks(x)
		return self.to_logits(x)




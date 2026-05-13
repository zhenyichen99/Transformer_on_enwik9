import torch
import numpy as np
from torch import nn
from data_util import enwik9, random_batches
from transformer import transformer
from train_test import random_epochs, train, test, generate

# hyperparameters
learning_rate = 0.0001
weight_decay = 1e-4
batch_size = 32
seq_len = 512

# pick accelerator
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'

# load the data
train_data, val_data, test_data = enwik9(num_train=int(90e7), num_val=int(5e7), num_test=int(4.75e7), device=device)
print('data loaded...')

# instantiate model into accelerator
model = transformer(emb_dim=256, num_heads=8, num_blocks=12, num_tokens=256, seq_len=seq_len, is_causal=True, device=device)
model.load_state_dict(torch.load('trained_models/transformer_2026-05-12_474k_batches', map_location=device, weights_only=True))
# trained on 474k batches so far
print('model created...') 

# run epochs with randomly sampled batches
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# random_epochs(train_data, val_data, model, optimizer, num_epochs=200, num_train_bats=1000, num_val_bats=50, batch_size=batch_size, seq_len=seq_len, gen=True)

# hold-out testing
# batches = random_batches(1000, test_data, batch_size=batch_size, seq_len=seq_len)
# test_loss = test(batches, model)
# bpb = test_loss/np.log(2)
# print(f'\ntest_loss: {test_loss:>13.6f}')
# print(f'bits_per_byte: {bpb:>9.6f}')

# generating text from hold-out test prompts
starts = np.random.randint(0, len(test_data)-seq_len, size=10)
for i, start in enumerate(starts):
	prompt = test_data[start:start+seq_len]
	for temp in [0.5, 0.8, 1.0]:
		print(f'\nPrompt {i+1}\ntemp: {temp:>3.1f}\n--------------------------------')
		generate(model, prompt, ctxt_len=seq_len, temp=temp)
		print('--------------------------------\n')
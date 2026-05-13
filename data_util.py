import torch
import numpy as np

def str_to_int_tensor(s):
	# turn a string to a 1D int tensor

	return torch.tensor(np.frombuffer(s.encode('utf-8'), dtype=np.uint8), dtype=torch.int)

def int_tensor_to_str(t):
	# turn a 1D int tensor to a string

	s = ''
	for n in t:
		s += chr(max(32, n))

	return s


def enwik9(num_train=int(90e6), num_val=int(5e6), num_test=int(5e6), path='data/enwik9', device='cpu'):
	# enwik9 has 997,520,893 characters
	# default takes the first 1/10 of enwik9 and split into train/val/test = 90/5/5

	with open(path) as file:
		train_data, val_data, test_data = map(str_to_int_tensor, map(file.read, (num_train, num_val, num_test)))

	return train_data.to(device), val_data.to(device), test_data.to(device)


def random_batch(data, batch_size, seq_len):
	# pick batch_size many random starting indices,
	# then slice off text chucks of length seq_len from data
	# return a pair of (batch_size, seq_len) tensors of torch.uint8
	# data: a 1D tensor of torch.uint8

	# randomly select starting indices
	starts = np.random.randint(0, len(data)-seq_len, size=batch_size)

	# slice and load the text chunks
	source_chunks = []
	target_chunks = []
	for i in starts:
		source_chunks.append(data[i:i+seq_len])
		target_chunks.append(data[i+1:i+1+seq_len])

	return torch.stack(source_chunks), torch.stack(target_chunks)


def random_batches(num_batches, data, batch_size, seq_len):
	# generate a length=num_batches list of random batches 

	batches = []
	for _ in range(num_batches):
		batches.append(random_batch(data, batch_size, seq_len))

	return batches







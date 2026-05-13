import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import matplotlib.pyplot as plt
import datetime
from data_util import random_batches, int_tensor_to_str

def train(batches, model, optimizer):

    model.train()
    loss_fn = nn.CrossEntropyLoss()
    num_batches = len(batches)
    train_loss = 0

    for x, y in batches:

        # propagate forward
        logits = model(x)
        loss = loss_fn(logits.transpose(2, 1), y.long())
        # print(loss.item())
        train_loss += loss.item()

        # backpropagate
        loss.backward()
        # for param in model.parameters():
        #     print(param.grad)
        optimizer.step()
        optimizer.zero_grad()

    # return avg train loss per batch
    train_loss /= num_batches
    return train_loss


def test(batches, model):

    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    num_batches = len(batches)
    test_loss = 0

    # testing
    with torch.no_grad():
        for x, y in batches:
            logits = model(x)
            test_loss += loss_fn(logits.transpose(2, 1), y.long()).item()

    # return avg test loss per batch
    test_loss /= num_batches
    return test_loss


def sample(logits, temp=0):
    # given logits, sample with given temperature

    if temp == 0:
        return logits.argmax()

    probs = F.softmax(logits/temp, dim=0)
    d = dist.Categorical(probs)

    return d.sample()

def generate(model, prompt, ctxt_len=512, gen_len=1024, temp=0):
    # prompt is a 1D tensor of tokens with length >= ctxt_len
    # ctxt_len should equal seq_len of model
    # gen_len is the number of additional characters to generate
    # temp is the sampling temperature

    assert len(prompt) >= ctxt_len, 'Length of prompt need to be at least ctxt_len.'

    print('\n\x1b[32m'+int_tensor_to_str(prompt)+'\x1b[0m', end='', flush=True)

    for _ in range(gen_len):

        x = prompt[-ctxt_len:][None]

        with torch.no_grad():
            logits = model(x)[0][-1]

        token = sample(logits, temp)[None]
        prompt = torch.cat((prompt, token))

        char = int_tensor_to_str(token)
        print(char, end='', flush=True)

    print('\n')

    return prompt


def random_epochs(train_data, val_data, model, optimizer, num_epochs, num_train_bats, num_val_bats, batch_size, seq_len, gen=False):

    if not gen:
        print(f'\nepoch | train_loss | val_loss ')

    for e in range(num_epochs):

        if gen:
            print(f'\nEpoch {e+1}\n--------------------------------')
        else:
            print(f'{e+1:<6}|', end='')

        train_batches = random_batches(num_train_bats, train_data, batch_size, seq_len)
        train_loss = train(train_batches, model, optimizer)

        if gen:
            print(f'train_loss: {train_loss:>10.6f}')
        else:
            print(f'{train_loss:>9.6f}   |', end='')

        val_batches = random_batches(num_val_bats, val_data, batch_size, seq_len)
        val_loss = test(val_batches, model)

        if gen:
            print(f'val_loss: {val_loss:>12.6f}')
            start = np.random.randint(0, len(val_data)-seq_len)
            prompt = val_data[start:start+seq_len]
            generate(model, prompt, ctxt_len=seq_len, temp=0.5)
            print('--------------------------------\n')
        else:
            print(f'{val_loss:>8.6f}')

        if e % 5 == 4:
            timestamp = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')[:19]
            save_as = 'trained_models/transformer_' + timestamp
            torch.save(model.state_dict(), save_as)
            print(f'model saved at epoch {e+1}')

    timestamp = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')[:19]
    save_as = 'trained_models/transformer_' + timestamp
    torch.save(model.state_dict(), save_as)
    print('final model saved')

        






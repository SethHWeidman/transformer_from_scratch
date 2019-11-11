# PyTorch
import torch
import torch.nn as nn

# transformer imports
from transformer_from_scratch.transformer import (
    Transformer1,
    Transformer2,
    Transformer3,
    Transformer4,
    Transformer5,
)

# source sequence length
S = 10

# target sequence length
T = 20

# batch size
N = 32

# feature size
# note: assumes tokens have already been passed through an embedding layer
E = 50

# number of heads
# necessary since feature size has to be divisible by the number of heads
H = 5

src1 = torch.rand(S, N, E)
tgt1 = torch.rand(T, N, E)

# vocab size
V = 20000

t = nn.Transformer(d_model=E, nhead=H)
t1 = Transformer1(d_model=E, nhead=H)

src2 = torch.randint(low=0, high=V, size=(T, N))
tgt2 = torch.randint(low=0, high=V, size=(T, N))

t2 = Transformer2(d_model=E, nhead=H, vocab_size=V, max_len=T)
t3 = Transformer3(d_model=E, nhead=H, vocab_size=V, max_len=T)
t4 = Transformer4(d_model=E, nhead=H, vocab_size=V, max_len=T)
t5 = Transformer5(d_model=E, nhead=H, vocab_size=V, max_len=T)


def test_transformer():
    assert (t(src1, tgt1).shape) == torch.Size([T, N, E])


def test_transformer_1():
    assert (t1(src1, tgt1).shape) == torch.Size([T, N, E])


def test_transformer_2():
    assert (t2(src2, tgt2).shape) == torch.Size([T, N, V])


def test_transformer_3():
    assert (t3(src2, tgt2).shape) == torch.Size([T, N, V])


def test_transformer_4():
    assert (t4(src2, tgt2).shape) == torch.Size([T, N, V])


def test_transformer_5():
    assert (t5(src2, tgt2).shape) == torch.Size([T, N, V])

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
    Transformer6,
)

from transformer_from_scratch.transformer import _generate_square_subsequent_mask

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
t6 = Transformer6(d_model=E, nhead=H, vocab_size=V, max_len=T)


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


# for sequence to sequence, every element of src can look at
# every element of src. But, while each element of trg can
# look at every element of src, it can only look at the future
# elements of trg

# To do a language model with this, on the other hand, each element of src
# can only look at past elements of src.

tgt_mask = _generate_square_subsequent_mask(T)


def test_transformer_6():
    assert (t6(src2, tgt2, tgt_mask=tgt_mask).shape) == torch.Size([T, N, V])

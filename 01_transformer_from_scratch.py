import torch
import torch.nn as nn


if __name__=='__main__':
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

    src = torch.rand(S, N, E)
    tgt = torch.rand(T, N, E)

    t = nn.Transformer(d_model = E,
                       nhead = H)

    out = t(src, tgt)
    print(out.shape)

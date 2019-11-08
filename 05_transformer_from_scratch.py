"""
Defines a "TransformerFull" class that includes both an embedding for the vocabulary and
a positional encoding.

Also includes custom TransformerEncoder and TransformerDecoder.
"""

import copy
import math

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init
from torch.nn.modules import ModuleList


def _get_clones(module: nn.Module, N: int) -> ModuleList:
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class PositionalEncoding(nn.Module):
    """
    Implements positional encoding as described in "Attention is All You Need"

    Expects input of size (N, T, E)

    Generates positional encoding of size (T, E), and adds this to each batch
    element.
    """

    def __init__(self, num_features: int, seq_len: int) -> None:
        super().__init__()

        # Encoding for each element is (seq_len x num_features)
        positional_encoding = torch.zeros(seq_len, num_features, requires_grad=False)

        # Generate position - list from 0 to seq_len
        # Reshape to (seq_len x 1)
        position = torch.arange(0, seq_len).unsqueeze(1).float()

        # These will be divided by
        #
        # (10000 ^ (i / num_features))
        #
        # where i is the dim
        #
        # So, we'll have one feature where the position is divided by 1, giving a
        # sine/cosine wave with frequency 2 * pi
        #
        # At the other extreme, we'll have a feature where the position is divided by
        # 10000, giving sine/cosine waves with frequency 2 * pi * 10000
        #
        # Another way of saying this is that this will be *multiplied* by
        # ((1/10000) ^ (i / num_features))
        #
        # or by
        #
        # exp ( log (1/10000) ^ (i / num_features) )
        #
        # or equivalently
        #
        # exp ( (i / num_features) * -log(10000) )
        div_term = torch.exp(
            (torch.arange(0, num_features, 2).float() / num_features)
            * math.log(1 / 10000)
        )
        # Now we alternate applying sine to these features vs. cosine
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        # Add a first dimension of size 1
        # [seq_len x num_features] -> [1 x seq_len x num_features]
        positional_encoding = positional_encoding.unsqueeze(0)

        # de-mean
        # due to all the cosine terms starting at 1, and the sine terms starting at
        # 0, the mean of these positional encodings is much greater than 0; adding
        # an embedding that is shifted like this seems sub optimal, so we'll
        # "de-mean" this matrix:
        positional_encoding = positional_encoding - positional_encoding.mean()

        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.positional_encoding


class TransformerEncoderCustom(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int) -> None:
        super(TransformerEncoderCustom, self).__init__()

        self.num_layers = num_layers
        self.layers = _get_clones(encoder_layer, num_layers)

    def forward(self, x: Tensor) -> None:

        out = x

        for i in range(self.num_layers):
            out = self.layers[i](out)

        return out


class TransformerDecoderCustom(nn.Module):
    def __init__(self, decoder_layer: nn.Module, num_layers: int) -> None:
        super(TransformerDecoderCustom, self).__init__()

        self.num_layers = num_layers
        self.layers = _get_clones(decoder_layer, num_layers)

    def forward(self, tgt: Tensor, memory: Tensor) -> None:

        out = tgt

        for i in range(self.num_layers):
            output = self.layers[i](tgt, memory)

        return out


class TransformerFull(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        vocab_size: int,
        max_len: int,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
    ) -> None:
        super(TransformerFull, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)

        self.encoder = TransformerEncoderCustom(encoder_layer, num_encoder_layers)
        self.decoder = TransformerDecoderCustom(decoder_layer, num_decoder_layers)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.output_bias = Parameter(torch.Tensor(vocab_size))
        self._init_bias()

    def _init_bias(self):
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L79-L84
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.embedding.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.output_bias, -bound, bound)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:

        embeddings = self.embedding(src)

        positional_encoding = self.positional_encoding(embeddings)

        memory = self.encoder(positional_encoding)

        tgt_embeddings = self.embedding(tgt)

        output = self.decoder(tgt_embeddings, memory)
        output = F.linear(output, self.embedding.weight, self.output_bias)

        return output


if __name__ == "__main__":
    # source sequence length
    S = 10

    # target sequence length
    T = 20

    # batch size
    N = 32

    # feature size
    E = 50

    # number of heads
    H = 5

    # vocab size
    V = 20000

    src = torch.randint(low=0, high=V, size=(N, T))
    tgt = torch.randint(low=0, high=V, size=(N, T))

    t = TransformerFull(d_model=E, nhead=H, vocab_size=V, max_len=T)
    out = t(src, tgt)
    print(out.shape)
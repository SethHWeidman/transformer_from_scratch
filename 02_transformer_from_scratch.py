"""
Defines a "TransformerCustom" class that breaks out the encoder and decoder
parts of the Transformer. 
Also illustrates the correct input shapes for a Transformer.
"""

import torch
import torch.nn as nn


class TransformerCustom(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
    ):
        super(TransformerCustom, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

    def forward(self, src, tgt):

        memory = self.encoder(src)

        output = self.decoder(tgt, memory)

        return output


if __name__ == "__main__":
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

    t = TransformerCustom(d_model=E, nhead=H)

    out = t(src, tgt)
    print(out.shape)

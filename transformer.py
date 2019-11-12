# Python standard library
import math
import copy

# PyTorch
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init
from torch.nn.modules import ModuleList
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm


class Transformer1(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
    ):
        super(Transformer1, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

    def forward(self, src: Tensor, tgt: Tensor) -> None:

        memory = self.encoder(src)

        output = self.decoder(tgt, memory)

        return output


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

        # Transpose to put sequence length in first position, batch size in second
        positional_encoding = positional_encoding.transpose(0, 1)

        # de-mean
        # due to all the cosine terms starting at 1, and the sine terms starting at
        # 0, the mean of these positional encodings is much greater than 0; adding
        # an embedding that is shifted like this seems sub optimal, so we'll
        # "de-mean" this matrix:
        positional_encoding = positional_encoding - positional_encoding.mean()

        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.positional_encoding


class Transformer2(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        vocab_size: int,
        max_len: int,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
    ) -> None:
        super(Transformer2, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

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


class Transformer3(Transformer2):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        vocab_size: int,
        max_len: int,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
    ) -> None:
        super(Transformer3, self).__init__(d_model, nhead, vocab_size, max_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)

        self.encoder = TransformerEncoderCustom(encoder_layer, num_encoder_layers)
        self.decoder = TransformerDecoderCustom(decoder_layer, num_decoder_layers)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.output_bias = Parameter(torch.Tensor(vocab_size))
        self._init_bias()


class TransformerEncoderLayerCustom(nn.Module):
    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout=0.1
    ) -> None:
        super(TransformerEncoderLayerCustom, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=0.1)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, src: Tensor) -> Tensor:

        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)

        src = self.norm1(src)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))

        src = src + self.dropout2(src2)

        src = self.norm2(src)

        return src


class TransformerDecoderLayerCustom(nn.Module):
    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout=0.1
    ) -> None:
        super(TransformerDecoderLayerCustom, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:

        tgt2 = self.self_attn(tgt, tgt, tgt)[0]

        tgt = tgt + self.dropout1(tgt2)

        tgt = self.norm1(tgt)

        tgt = self.multihead_attn(tgt, memory, memory)[0]

        tgt = tgt + self.dropout(tgt2)

        tgt = self.norm2(tgt)

        tgt = self.linear2(self.dropout(F.relu(self.linear1(tgt))))

        tgt = tgt + self.dropout3(tgt2)

        tgt = self.norm3(tgt)

        return tgt


class Transformer4(Transformer3):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        vocab_size: int,
        max_len: int,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
    ) -> None:
        super(Transformer4, self).__init__(d_model, nhead, vocab_size, max_len)

        encoder_layer = TransformerEncoderLayerCustom(d_model=d_model, nhead=nhead)
        decoder_layer = TransformerDecoderLayerCustom(d_model=d_model, nhead=nhead)

        self.encoder = TransformerEncoderCustom(encoder_layer, num_encoder_layers)
        self.decoder = TransformerDecoderCustom(decoder_layer, num_decoder_layers)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.output_bias = Parameter(torch.Tensor(vocab_size))
        self._init_bias()


class MultiheadAttentionCustom(nn.Module):
    def __init__(
        self, d_model: int, nhead: int, dropout: float = 0,
    ):
        super(MultiheadAttentionCustom, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.head_dim = d_model // nhead

        # we will not assume that the query, key, and value have the same
        # dimension, even though that will often be true in practice
        self.kdim = d_model
        self.vdim = d_model

        self.q_proj_weight = Parameter(torch.Tensor(d_model, d_model))
        self.k_proj_weight = Parameter(torch.Tensor(d_model, d_model))
        self.v_proj_weight = Parameter(torch.Tensor(d_model, d_model))

        self.q_bias = Parameter(torch.empty(d_model))
        self.k_bias = Parameter(torch.empty(d_model))
        self.v_bias = Parameter(torch.empty(d_model))

        self.out_proj = Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.q_proj_weight)
        init.xavier_uniform_(self.k_proj_weight)
        init.xavier_uniform_(self.v_proj_weight)

        init.constant_(self.q_bias, 0.0)
        init.constant_(self.k_bias, 0.0)
        init.constant_(self.v_bias, 0.0)

    def forward(self, query: Tensor, key: Tensor, value: Tensor):

        tgt_len, batch_size, d_model = query.size()
        # get projected versions of all three Tensors
        q_proj = F.linear(query, self.q_proj_weight, self.q_bias)
        k_proj = F.linear(key, self.k_proj_weight, self.k_bias)
        v_proj = F.linear(value, self.v_proj_weight, self.v_bias)

        # transpose q from [T, N, E] to [N * H, T, E / H]
        q_proj = (
            q_proj.contiguous()  # unnecessary
            .view(tgt_len, batch_size * self.nhead, self.head_dim)
            .transpose(0, 1)
        )

        # transpose k so that last two dimensions are [src_len, d_model / nhead]
        k_proj = (
            k_proj.contiguous()  # unnecessary
            .view(-1, batch_size * self.nhead, self.head_dim)
            .transpose(0, 1)
        )

        # transpose v so that last two dimensions are [src_len, d_model / nhead]
        v_proj = (
            v_proj.contiguous()  # unnecessary
            .view(-1, batch_size * self.nhead, self.head_dim)
            .transpose(0, 1)
        )

        q_proj *= float(self.head_dim) ** -0.5

        # "T" - works because k is three dimensional
        src_len = k_proj.size(1)

        # compute attention output weights
        # q_proj shape: batch_size * num_heads, tgt_len, d_model / nhead
        # k_proj.transpose(1, 2) shape: batch_size * num_heads, src_len, d_model / nhead
        # shape: batch_size * num_heads, tgt_len, src_len
        attn_output_weights = torch.bmm(q_proj, k_proj.transpose(1, 2))

        # apply softmax so that values along src_len dimension add to 1
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)

        # dropout
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout)

        # use these weights to compute a weighted average of the values
        # attn_output shape: batch_size * num_heads x tgt_len x d_model / nhead
        attn_output = torch.bmm(attn_output_weights, v_proj)

        # use these weights to compute a weighted average of the values
        attn_output = (
            attn_output.transpose(0, 1)  # puts tgt_len as first dimension
            .contiguous()  # unnecessary
            .view(tgt_len, batch_size, d_model)  # reshape to size of output
        )

        attn_output = self.out_proj(attn_output)

        return (attn_output,)


class TransformerEncoderLayerCustom2(TransformerEncoderLayerCustom):
    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout=0.1
    ) -> None:
        super(TransformerEncoderLayerCustom2, self).__init__(d_model, nhead)
        self.self_attn = MultiheadAttentionCustom(d_model, nhead, dropout=0.1)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)


class TransformerDecoderLayerCustom2(TransformerDecoderLayerCustom):
    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout=0.1
    ) -> None:
        super(TransformerDecoderLayerCustom2, self).__init__(d_model, nhead)
        self.self_attn = MultiheadAttentionCustom(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttentionCustom(d_model, nhead, dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)


class Transformer5(Transformer4):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        vocab_size: int,
        max_len: int,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
    ) -> None:
        super(Transformer5, self).__init__(
            d_model, nhead, vocab_size, max_len, num_encoder_layers, num_decoder_layers
        )

        encoder_layer = TransformerEncoderLayerCustom2(d_model=d_model, nhead=nhead)
        decoder_layer = TransformerDecoderLayerCustom2(d_model=d_model, nhead=nhead)

        self.encoder = TransformerEncoderCustom(encoder_layer, num_encoder_layers)
        self.decoder = TransformerDecoderCustom(decoder_layer, num_decoder_layers)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.output_bias = Parameter(torch.Tensor(vocab_size))
        self._init_bias()


class TransformerDecoderCustom2(TransformerDecoderCustom):
    def __init__(self, decoder_layer: nn.Module, num_layers: int) -> None:
        super(TransformerDecoderCustom2, self).__init__(decoder_layer, num_layers)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor = None,
        tgt_key_padding_mask: Tensor = None,
        memory_mask: Tensor = None,
        memory_key_padding_mask: Tensor = None,
    ) -> None:

        out = tgt

        for i in range(self.num_layers):
            output = self.layers[i](
                tgt,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_mask=memory_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        return out


class TransformerEncoderCustom2(TransformerEncoderCustom):
    def __init__(self, encoder_layer: nn.Module, num_layers: int) -> None:
        super(TransformerEncoderCustom2, self).__init__(encoder_layer, num_layers)

    def forward(
        self, x: Tensor, attn_mask: Tensor = None, key_padding_mask: Tensor = None
    ) -> None:

        out = x

        for i in range(self.num_layers):
            out = self.layers[i](
                out, attn_mask=attn_mask, key_padding_mask=key_padding_mask
            )

        return out


class TransformerDecoderLayerCustom3(TransformerDecoderLayerCustom2):
    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout=0.1
    ) -> None:
        super(TransformerDecoderLayerCustom3, self).__init__(
            d_model, nhead, dim_feedforward, dropout
        )
        self.self_attn = MultiheadAttentionCustom2(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttentionCustom2(d_model, nhead, dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor = None,
        tgt_key_padding_mask: Tensor = None,
        memory_mask: Tensor = None,
        memory_key_padding_mask: Tensor = None,
    ) -> Tensor:

        tgt2 = self.self_attn(tgt, tgt, tgt, tgt_mask, tgt_key_padding_mask)[0]

        tgt = tgt + self.dropout1(tgt2)

        tgt = self.norm1(tgt)

        tgt = self.multihead_attn(
            tgt, memory, memory, memory_mask, memory_key_padding_mask
        )[0]

        tgt = tgt + self.dropout(tgt2)

        tgt = self.norm2(tgt)

        tgt = self.linear2(self.dropout(F.relu(self.linear1(tgt))))

        tgt = tgt + self.dropout3(tgt2)

        tgt = self.norm3(tgt)

        return tgt


class TransformerEncoderLayerCustom3(TransformerEncoderLayerCustom2):
    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout=0.1
    ) -> None:
        super(TransformerEncoderLayerCustom3, self).__init__(
            d_model, nhead, dim_feedforward, dropout
        )
        self.self_attn = MultiheadAttentionCustom2(d_model, nhead, dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(
        self, src: Tensor, attn_mask: Tensor = None, key_padding_mask: Tensor = None
    ) -> Tensor:

        src2 = self.self_attn(
            src, src, src, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )[0]
        src = src + self.dropout1(src2)

        src = self.norm1(src)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))

        src = src + self.dropout2(src2)

        src = self.norm2(src)

        return src


class MultiheadAttentionCustom2(MultiheadAttentionCustom):
    def __init__(
        self, d_model: int, nhead: int, dropout: float = 0.0,
    ):
        super(MultiheadAttentionCustom2, self).__init__(d_model, nhead, dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor = None,
        key_padding_mask: Tensor = None,
    ) -> Tensor:

        tgt_len, batch_size, d_model = query.size()
        # get projected versions of all three Tensors
        q_proj = F.linear(query, self.q_proj_weight, self.q_bias)
        k_proj = F.linear(key, self.k_proj_weight, self.k_bias)
        v_proj = F.linear(value, self.v_proj_weight, self.v_bias)

        # transpose q from [T, N, E] to [N * H, T, E / H]
        q_proj = (
            q_proj.contiguous()  # unnecessary
            .view(tgt_len, batch_size * self.nhead, self.head_dim)
            .transpose(0, 1)
        )

        # transpose k so that last two dimensions are [src_len, d_model / nhead]
        k_proj = (
            k_proj.contiguous()  # unnecessary
            .view(-1, batch_size * self.nhead, self.head_dim)
            .transpose(0, 1)
        )

        # transpose v so that last two dimensions are [src_len, d_model / nhead]
        v_proj = (
            v_proj.contiguous()  # unnecessary
            .view(-1, batch_size * self.nhead, self.head_dim)
            .transpose(0, 1)
        )

        q_proj *= float(self.head_dim) ** -0.5

        # "T" - works because k is three dimensional
        src_len = k_proj.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == batch_size
            assert key_padding_mask.size(1) == src_len

        # compute attention output weights
        # q_proj shape: batch_size * num_heads, tgt_len, d_model / nhead
        # k_proj.transpose(1, 2) shape: batch_size * num_heads, src_len, d_model / nhead
        # shape: batch_size * num_heads, tgt_len, src_len
        attn_output_weights = torch.bmm(q_proj, k_proj.transpose(1, 2))

        # attn_mask is two dimensional Tensor of shape tgt_len x src_len
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_output_weights += attn_mask

        # key padding mask is two dimensional Tensor of shape batch size, src sequence length
        if key_padding_mask is not None:
            # need to "unravel" this to get batch size and src_len isolated
            attn_output_weights = attn_output_weights.view(
                batch_size, nhead, tgt_len, src_len
            )

            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
            attn_output_weights = attn_output_weights.view(
                batch_size * nhead, tgt_len, src_len
            )

        # apply softmax so that values along src_len dimension add to 1
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)

        # dropout
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout)

        # use these weights to compute a weighted average of the values
        # attn_output shape: batch_size * num_heads x tgt_len x d_model / nhead
        attn_output = torch.bmm(attn_output_weights, v_proj)

        # use these weights to compute a weighted average of the values
        attn_output = (
            attn_output.transpose(0, 1)  # puts tgt_len as first dimension
            .contiguous()  # unnecessary
            .view(tgt_len, batch_size, d_model)  # reshape to size of output
        )

        attn_output = self.out_proj(attn_output)

        return attn_output


class Transformer6(Transformer5):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        vocab_size: int,
        max_len: int,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
    ) -> None:
        super(Transformer6, self).__init__(d_model, nhead, vocab_size, max_len)

        encoder_layer = TransformerEncoderLayerCustom3(d_model=d_model, nhead=nhead)
        decoder_layer = TransformerDecoderLayerCustom3(d_model=d_model, nhead=nhead)

        self.encoder = TransformerEncoderCustom2(encoder_layer, num_encoder_layers)
        self.decoder = TransformerDecoderCustom2(decoder_layer, num_decoder_layers)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.output_bias = Parameter(torch.Tensor(vocab_size))
        self._init_bias()

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor = None,
        tgt_mask: Tensor = None,
        memory_mask: Tensor = None,
        src_key_padding_mask: Tensor = None,
        tgt_key_padding_mask: Tensor = None,
        memory_key_padding_mask: Tensor = None,
    ) -> Tensor:

        embeddings = self.embedding(src)

        positional_encoding = self.positional_encoding(embeddings)

        memory = self.encoder(
            positional_encoding,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )

        tgt_embeddings = self.embedding(tgt)

        output = self.decoder(
            tgt_embeddings,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        output = F.linear(output, self.embedding.weight, self.output_bias)

        return output


def _generate_square_subsequent_mask(sz: int):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def _get_clones(module: nn.Module, N: int) -> ModuleList:
    return ModuleList([copy.deepcopy(module) for i in range(N)])

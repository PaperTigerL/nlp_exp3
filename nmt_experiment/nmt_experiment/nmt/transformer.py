import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionEmbedding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        # shape: [max_len, 1]
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)

        # shape: [d_model//2, ]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )

        # shape: [max_len, 1, d_model]
        pe = torch.empty([max_len, 1, d_model])
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, input):
        # shape of input: [seq_len, batch_size, d_model]
        return self.pe[:input.shape[0]]


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, dropout):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        self.WO = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

        self.scale = self.d_k ** -0.5


    def forward(self, query, key, value, key_padding_mask = None, attn_mask = None):
        # shape of query: [tgt_len, batch_size, d_model]
        # shape of key: [src_len, batch_size, d_model]
        # shape of value: [src_len, batch_size, d_model]
        tgt_len = query.shape[0]
        batch_size = query.shape[1]
        src_len = key.shape[0]

        query = self.WQ(query)
        key = self.WK(key)
        value = self.WV(value)

        query = query * self.scale

        # shape: [batch_size * num_heads, tgt_len, d_k]
        query = query.view(tgt_len, batch_size * self.num_heads, self.d_k).transpose(0, 1)

        # shape: [batch_size * num_heads, src_len, d_k]
        key = key.view(src_len, batch_size * self.num_heads, self.d_k).transpose(0, 1)

        # shape: [batch_size * num_heads, src_len, d_v]
        value = value.view(src_len, batch_size * self.num_heads, self.d_v).transpose(0, 1)

        # attn_weights: [batch_size * num_heads, tgt_len, src_len]
        attn_weights = torch.bmm(query, key.transpose(1, 2))

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights = attn_weights + attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(batch_size, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
            attn_weights = attn_weights.view(batch_size * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # shape: (batch_size * num_heads, tgt_len, d_v)
        output = torch.bmm(attn_weights, value)

        # shape: (tgt_len, batch_size, d_model)
        output = output.transpose(0, 1).contiguous().view(tgt_len, batch_size, self.d_model)

        output = self.WO(output)
        return output


class PositionwiseFeedforward(nn.Module):

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()

        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input):
        # shape: [seq_len, batch_size, d_ff]
        input = self.W1(input)
        
        input = F.relu(input)
        input = self.dropout(input)

        input = self.W2(input)
        return input


class EncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)

        self.self_attention_layer_norm = nn.LayerNorm(d_model)
        self.feed_forward_layer_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input, padding_mask = None):
        residual = input
        input = self.self_attention(input, input, input, key_padding_mask=padding_mask)
        input = self.dropout(input)
        input = input + residual
        input = self.self_attention_layer_norm(input)

        residual = input
        input = self.feed_forward(input)
        input = self.dropout(input)
        input = input + residual
        input = self.feed_forward_layer_norm(input)
        return input


class Encoder(nn.Module):

    def __init__(
        self, 
        d_model, 
        num_layers, 
        num_heads, 
        d_ff,
        dropout,
        vocab_size, 
        max_len = 1024, 
    ):
        super().__init__()

        self.position_embedding = PositionEmbedding(d_model, max_len)

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model, 
                    num_heads, 
                    d_ff,
                    dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(p=dropout)

        self.scale = math.sqrt(d_model)
        self.padding_idx = 1

    def forward(self, input):

        # shape: [batch_size, seq_len]
        encoder_padding_mask = input.eq(self.padding_idx)

        # shape: [batch_size, seq_len, d_model]
        input = self.token_embedding(input) * self.scale

        # shape: [seq_len, batch_size, d_model]
        input = input.transpose(0, 1)

        input = input + self.position_embedding(input)

        for layer in self.layers:
            input = layer(input, encoder_padding_mask)

        return input, encoder_padding_mask


class DecoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)

        self.self_attention_layer_norm = nn.LayerNorm(d_model)
        self.cross_attention_layer_norm = nn.LayerNorm(d_model)
        self.feed_forward_layer_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self, 
        input, 
        encoder_out, 
        self_attn_mask, 
        self_attn_padding_mask, 
        encoder_padding_mask
    ):
        residual = input
        input = self.self_attention(
            query=input, 
            key=input, 
            value=input, 
            key_padding_mask=self_attn_padding_mask,
            attn_mask=self_attn_mask
        )
        input = self.dropout(input)
        input = input + residual
        input = self.self_attention_layer_norm(input)

        residual = input
        input = self.cross_attention(
            query=input, 
            key=encoder_out, 
            value=encoder_out, 
            key_padding_mask=encoder_padding_mask
        )
        input = self.dropout(input)
        input = input + residual
        input = self.cross_attention_layer_norm(input)

        residual = input
        input = self.feed_forward(input)
        input = self.dropout(input)
        input = input + residual
        input = self.feed_forward_layer_norm(input)
        return input


class Decoder(nn.Module):

    def __init__(
        self, 
        d_model, 
        num_layers, 
        num_heads, 
        d_ff,
        dropout,
        vocab_size, 
        max_len = 1024, 
    ):
        super().__init__()

        self.position_embedding = PositionEmbedding(d_model, max_len)

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model, 
                    num_heads, 
                    d_ff,
                    dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.linear = nn.Linear(d_model, vocab_size)

        self.scale = math.sqrt(d_model)
        self.padding_idx = 1
        self._future_mask = torch.empty(0)

    def forward(
        self, 
        input, 
        encoder_out, 
        encoder_padding_mask
    ):
        # shape: [batch_size, seq_len]
        self_attn_padding_mask = input.eq(self.padding_idx)

        # shape: [batch_size, seq_len, d_model]
        input = self.token_embedding(input) * self.scale

        # shape: [seq_len, batch_size, d_model]
        input = input.transpose(0, 1)

        input = input + self.position_embedding(input)

        for layer in self.layers:
            self_attn_mask = self.buffered_future_mask(input)

            input = layer(
                input, 
                encoder_out, 
                self_attn_mask, 
                self_attn_padding_mask, 
                encoder_padding_mask
            )
        
        # shape: [batch_size, seq_len, d_model]
        input = input.transpose(0, 1)

        # input: [batch_size, seq_len, vocab_size]
        input = self.linear(input)
        return input


    def buffered_future_mask(self, tensor):
        dim = tensor.shape[0]
        if (
            self._future_mask.shape[0] == 0
            or self._future_mask.shape[0] < dim
        ):
            self._future_mask = torch.triu(
                torch.empty([dim, dim], dtype=torch.float).fill_(float("-inf")), 1
            )
        
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]


class Transformer(nn.Module):
    def __init__(
        self, 
        encoder, 
        decoder,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src_input, tgt_input):
        encoder_out, encoder_padding_mask = self.encoder(src_input)
        output = self.decoder(tgt_input, encoder_out, encoder_padding_mask)
        return output

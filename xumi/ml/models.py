import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import Linear
from torch.nn import functional as F


def gen_trg_mask(length, device):
    return torch.triu(
        torch.ones(length, length, device=device) * float("-inf"), diagonal=1
    )


def create_padding_mask(tensor, pad_idx):
    padding_mask = (tensor == pad_idx).transpose(0, 1)

    return padding_mask


def masked_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor, pad_idx):
    mask = y_true != pad_idx
    y_true = torch.masked_select(y_true, mask)
    y_pred = torch.masked_select(y_pred, mask)

    acc = (y_true == y_pred).double().mean()

    return acc


class PositionalEncoding(nn.Module):
    #  https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """

        x = x + self.pe[: x.size(0)]

        return self.dropout(x)


class TokenEmbedding(nn.Module):
    #  https://pytorch.org/tutorials/beginner/translation_transformer.html
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Seq2Seq(pl.LightningModule):
    def __init__(
        self,
        out_vocab_size,
        pad_idx,
        channels=256,
        dropout=0.1,
        lr=1e-4,
    ):
        super().__init__()

        self.lr = lr
        self.pad_idx = pad_idx
        self.dropout = dropout
        self.out_vocab_size = out_vocab_size

        self.embeddings = TokenEmbedding(
            vocab_size=self.out_vocab_size, emb_size=channels
        )

        self.pos_encoder = PositionalEncoding(d_model=channels, dropout=dropout)

        self.transformer = torch.nn.Transformer(
            d_model=channels,
            nhead=4,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=1024,
            dropout=dropout,
        )

        self.linear = Linear(channels, out_vocab_size)

        self.do = nn.Dropout(p=self.dropout)

    def init_weights(self) -> None:
        init_range = 0.1
        self.embeddings.weight.data.uniform_(-init_range, init_range)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-init_range, init_range)

    def encode_src(self, src):
        src = src.permute(1, 0)

        src_pad_mask = create_padding_mask(src, self.pad_idx)

        src = self.embeddings(src)

        src = self.pos_encoder(src)

        src = self.transformer.encoder(src, src_key_padding_mask=src_pad_mask)

        src = self.pos_encoder(src)

        return src

    def decode_trg(self, trg, memory):
        trg = trg.permute(1, 0)

        out_sequence_len, batch_size = trg.size(0), trg.size(1)

        trg_pad_mask = create_padding_mask(trg, self.pad_idx)

        trg = self.embeddings(trg)

        trg = self.pos_encoder(trg)

        trg_mask = gen_trg_mask(out_sequence_len, trg.device)

        out = self.transformer.decoder(
            tgt=trg, memory=memory, tgt_mask=trg_mask, tgt_key_padding_mask=trg_pad_mask
        )

        out = out.permute(1, 0, 2)

        out = self.linear(out)

        return out

    def forward(self, x):
        src, trg = x

        src = self.encode_src(src)

        out = self.decode_trg(trg=trg, memory=src)

        return out

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, name="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, name="valid")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, name="test")

    def _step(self, batch, batch_idx, name="train"):
        src, trg = batch

        trg_in, trg_out = trg[:, :-1], trg[:, 1:]

        y_hat = self((src, trg_in))

        y_hat = y_hat.view(-1, y_hat.size(2))
        y = trg_out.contiguous().view(-1)

        loss = F.cross_entropy(y_hat, y, ignore_index=self.pad_idx)

        _, predicted = torch.max(y_hat, 1)
        acc = masked_accuracy(y, predicted, pad_idx=self.pad_idx)

        self.log(f"{name}_loss", loss)
        self.log(f"{name}_acc", acc)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    n_classes = 100

    source = torch.randint(low=0, high=n_classes, size=(20, 16))
    target = torch.randint(low=0, high=n_classes, size=(20, 32))

    s2s = Seq2Seq(out_vocab_size=n_classes, pad_idx=0)

    out = s2s((source, target))
    print(out.size())
    print(out)

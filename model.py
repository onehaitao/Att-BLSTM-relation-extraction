#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Att_BLSTM(nn.Module):
    def __init__(self, word_vec, class_num, config):
        super().__init__()
        self.word_vec = word_vec
        self.class_num = class_num

        # hyper parameters and others
        self.max_len = config.max_len
        self.word_dim = config.word_dim
        self.hidden_size = config.hidden_size
        self.layers_num = config.layers_num
        self.emb_dropout_value = config.emb_dropout
        self.lstm_dropout_value = config.lstm_dropout
        self.linear_dropout_value = config.linear_dropout

        # net structures and operations
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=self.word_vec,
            freeze=False,
        )
        self.lstm = nn.LSTM(
            input_size=self.word_dim,
            hidden_size=self.hidden_size,
            num_layers=self.layers_num,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=True,
        )
        self.tanh = nn.Tanh()
        self.emb_dropout = nn.Dropout(self.emb_dropout_value)
        self.lstm_dropout = nn.Dropout(self.lstm_dropout_value)
        self.linear_dropout = nn.Dropout(self.linear_dropout_value)

        self.att_weight = nn.Parameter(torch.randn(1, self.hidden_size, 1))
        self.dense = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.class_num,
            bias=True
        )

        # initialize weight
        init.xavier_normal_(self.dense.weight)
        init.constant_(self.dense.bias, 0.)

    def lstm_layer(self, x, mask):
        lengths = torch.sum(mask.gt(0), dim=-1)
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h, (_, _) = self.lstm(x)
        h, _ = pad_packed_sequence(h, batch_first=True, padding_value=0.0, total_length=self.max_len)
        h = h.view(-1, self.max_len, 2, self.hidden_size)
        h = torch.sum(h, dim=2)  # B*L*H
        return h

    def attention_layer(self, h, mask):
        att_weight = self.att_weight.expand(mask.shape[0], -1, -1)  # B*H*1
        att_score = torch.bmm(self.tanh(h), att_weight)  # B*L*H  *  B*H*1 -> B*L*1

        # mask, remove the effect of 'PAD'
        mask = mask.unsqueeze(dim=-1)  # B*L*1
        att_score = att_score.masked_fill(mask.eq(0), float('-inf'))  # B*L*1
        att_weight = F.softmax(att_score, dim=1)  # B*L*1

        reps = torch.bmm(h.transpose(1, 2), att_weight).squeeze(dim=-1)  # B*H*L *  B*L*1 -> B*H*1 -> B*H
        reps = self.tanh(reps)  # B*H
        return reps

    def forward(self, data):
        token = data[:, 0, :].view(-1, self.max_len)
        mask = data[:, 1, :].view(-1, self.max_len)
        emb = self.word_embedding(token)  # B*L*word_dim
        emb = self.emb_dropout(emb)
        h = self.lstm_layer(emb, mask)  # B*L*H
        h = self.lstm_dropout(h)
        reps = self.attention_layer(h, mask)  # B*reps
        reps = self.linear_dropout(reps)
        logits = self.dense(reps)
        return logits

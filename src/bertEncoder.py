import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
import itertools
import numpy as np

from transformers import *

class BertEncoder(nn.Module):
    def __init__(selfi, bertpath='bert-base-cased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(bertpath)
        self.bert.train()
        
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        self.maskSeed = nn.Parameter(torch.FloatTensor([0,1]))
        self.padding_idx = 0

    def forward(self, xs, embedW, padding_idx):
        padding_idx = self.padding_idx # this is dummy argument

        # xs: [[0,1,2], [0,1,2], [0,1,2]]...
        lens = [len(x) for x in xs]
        maxL = max(lens)

        # truncate exceeds
        if self.bert.config.max_position_embeddings < maxL:
            xs = [x[:self.bert.config.max_position_embeddings] for x in xs]
            maxL = self.bert.config.max_position_embeddings

        xs_pad = [x + [padding_idx]*(maxL-xl) for x, xl in zip(xs, lens)]

        attention_mask = [0 if x==padding_idx else 1 for x_pad in xs_pad for x in x_pad]
        attention_mask = self.maskSeed[attention_mask]
        attention_mask = attention_mask.view(len(xs_pad), -1)

        ems = embedW[xs_pad,:]

        ys, zs = self.bert.forward(inputs_embeds=ems, attention_mask=attention_mask)

        # should pad parts of zs be masked with -inf?
        #print(torch.var(zs, dim=0).sum())
        return zs, ys

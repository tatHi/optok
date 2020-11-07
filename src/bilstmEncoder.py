import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
import itertools

class BiLSTMEncoder(nn.Module):
    def __init__(self, embedSize, hidSize):
        super().__init__()
        self.bilstm = nn.LSTM(embedSize, hidSize, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidSize*2, hidSize)
        self.mode = 'max'
        
        self.hidSize = hidSize

    def forward(self, xs, embedW, padding_idx):
        # xs: [[0,1,2], [0,1,2], [0,1,2]]...
        lens = [len(x) for x in xs]
        maxL = max(lens)
        xs_pad = [x + [padding_idx]*(maxL-xl) for x, xl in zip(xs, lens)]

        ems = embedW[xs_pad,:]

        packed=torch.nn.utils.rnn.pack_padded_sequence(ems, lengths=lens, batch_first=True, enforce_sorted=False)
        ys, hs = self.bilstm(packed)
        ys, _  = torch.nn.utils.rnn.pad_packed_sequence(ys, batch_first=True, padding_value=float('-inf'))

        if self.mode=='max':
            zs = ys.permute(0,2,1)
            zs = F.max_pool1d(zs, kernel_size=zs.shape[2])
            zs = zs.view(len(xs), -1)
        elif self.mode=='last':
            # TODO: implement
            pass
        
        zs = torch.tanh(self.linear(zs))
        
        return zs, ys


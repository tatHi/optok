import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

class AttentionEncoder(nn.Module):
    def __init__(self, embedSize, hidSize):
        super().__init__()
        self.attnLinear = nn.Linear(embedSize, 1)
        self.linear = nn.Linear(embedSize, hidSize)
        self.maskSeed = nn.Parameter(torch.tensor([0, float('-inf')]), requires_grad=False)

    def forward(self, xs, embedW, padding_idx):
        # attention
        #mask = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=-1)
        #mask = (mask!=-1).unsqueeze(dim=2)

        padding_idx = -1

        lens = [len(x) for x in xs]
        maxL = max(lens)
        xs_pad = [x + [padding_idx]*(maxL-xl) for x, xl in zip(xs, lens)]
        mask = self.maskSeed[[[1 if x==padding_idx else 0 for x in xs] for xs in xs_pad],]
        mask = mask.view(len(xs), maxL, 1)
        ems = embedW[xs_pad,:] # idx -1 is obtained as pad but no effect due to mask

        attn = self.attnLinear(ems)
        attn = attn+mask
        attn = F.softmax(attn.squeeze()).unsqueeze(dim=2)

        # apply
        hid = ems * attn
        hid = hid.sum(dim=1)

        hid = F.tanh(self.linear(hid))
        return hid, ems+mask

if __name__=='__main__':
    bl = AttentionEncoder(3,5)
    xs = [[0,1,2],
          [1,2,3,4]]
    embedW = torch.rand(6,3)
    print(xs)
    print(embedW)
    print(torch.mean(embedW[xs[0]], dim=0))
    print(torch.mean(embedW[xs[1]], dim=0))
    print(bl(xs, embedW, 5))

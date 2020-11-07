import torch
import sys
sys.path.append('../src/')
import optok
import bilstmEncoder
from multigram import lm

text = [line.strip() for line in open('../data/train.text')]
mlm = lm.MultigramLM(data=text)
embed = torch.nn.Embedding(len(mlm.vocab),5)
bilstm = bilstmEncoder.BiLSTMEncoder(5, 3)

ot = optok.OpTok(mlm=mlm,
                 embed=embed,
                 encoder=bilstm,
                 m=3,
                 n=3,
                 topK=8,
                 samplingMode='soft',
                 ffbsMode=False,
                 selectMode='sampling',
                 lam=0.2,
                 tau=0.1,
                 mTest=1,
                 nTest=1,
                 samplingModeTest='top',
                 lamTest=1.0,
                 tauTest=0.01)

vss, uniLoss, nbests, attn, yss, hss = ot.forward(text)
print('vss')
print(vss)
print('uniLoss')
print(uniLoss)
print('nbests')
print(nbests)
print('attn')
print(attn)
print('yss')
print(yss)
print('hss')
print(hss)

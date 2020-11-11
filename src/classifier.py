import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn


import sys
sys.path.append('../../optok/src')
import optok

class Classifier(nn.Module):
    def __init__(self, mlm, 
                       embedSize, 
                       hidSize, 
                       labelSize, 
                       m, 
                       n, 
                       topK, 
                       lam,
                       selectMode,
                       encoderType, 
                       dropoutRate=0.5,
                       useIndividualEmbed=False):
        super().__init__()
        
        useIndividualEmbed = useIndividualEmbed

        # prepare embed
        self.lmEmbed = nn.Embedding(len(mlm.vocab), embedSize)
        if useIndividualEmbed:
            # if use individual embed is true, the model use different embedding 
            # for the language model and the classifier
            self.encEmbed = nn.Embedding(len(mlm.vocab), embedSize)

        # prepare encoder
        if encoderType=='bilstm':
            import bilstmEncoder
            encoder = bilstmEncoder.BiLSTMEncoder(embedSize, hidSize)
        elif encoderType=='bert':
            import bertEncoder
            encoder = bertEncoder.BertEncoder()
        else:
            print('encoer should be bilstm or bert')
            exit()
        
        self.ot = optok.OpTok(mlm, 
                              lmEmbed=self.lmEmbed,
                              encEmbed=self.encEmbed if useIndividualEmbed else self.lmEmbed, 
                              encoder=encoder, 
                              m=m,
                              n=n,
                              topK=topK,
                              selectMode=selectMode,
                              lam=lam)
        if encoderType=='bert':
            self.ot.bertMode = True

        self.dropout = nn.Dropout(p=dropoutRate)
        self.linear = nn.Linear(hidSize, labelSize)
        
        self.criterion = nn.CrossEntropyLoss()

    def calcScores(self, vs):
        vs = self.dropout(vs)
        scores = self.linear(vs)
        return scores

    def forward(self, lines, labels=None, uniLossWeight=0.0):
        vs, uniLoss, nbests, attn, yss, hss = self.ot(lines)
        scores = self.calcScores(vs)
        if labels is not None:
            clLoss = self.criterion(scores, labels)
            loss = clLoss + uniLossWeight*uniLoss
            return loss
        else:
            return scores, uniLoss

    def forwardWithGivenSegmentation(self, xss):
        vs = self.ot.encode([xss], len(xss))[0]
        return self.calcScores(vs)


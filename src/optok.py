import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
import itertools
import numpy as np
import sys
import unigramNLM
from multigram import mdp
from time import time

class OpTok(nn.Module):
    def __init__(self, mlm, 
                       lmEmbed,
                       encEmbed, 
                       encoder, 
                       m, 
                       n, 
                       topK, 
                       selectMode='sampling',
                       lam=0.2,
                       mTest=1,
                       nTest=1,
                       lamTest=1.0):
        super().__init__()
        self.mlm = mlm
        self.lmEmbed = lmEmbed
        self.encEmbed = encEmbed # encEmbed.shape should be same as lmEmbed.shape
        self.encoder = encoder
        self.m = m
        self.n = n

        self.lam = lam
        self.selectMode = selectMode

        # set parameter for test
        self.mTest = mTest
        self.nTest = nTest
        self.lamTest = lamTest
        ###

        vocabSize = len(mlm.vocab) #+ 1
        embedSize = self.lmEmbed.weight.shape[1]

        if '<unk>' in mlm.vocab:
            self.unkCharIdx = mlm.piece_to_id('<unk>') 
        elif '[UNK]' in mlm.vocab:
            self.unkCharIdx = mlm.word2id['[UNK]']
        else:
            # when using model dumped by older version, the model does not have unk token.
            # we use the last index as unkidx at that time.
            vocabSize += 1
            self.unkCharIdx = vocabSize-1
        
        self.minfPaddingIdx = vocabSize
        self.zeroPaddingIdx = vocabSize+1

        self.nlm = unigramNLM.UnigramNLM(vocabSize, embedSize, unkIdx=self.unkCharIdx)

        self.topK = topK

        self.CACHE_log_theta = None
        self.CACHE_log_smoothed_theta = None
        self.CACHE_vocab = None

        self.bertMode = False

    def encode(self,  xss, m):
        # encode
        xss_wo_pad = [[x for x in xs if x[0]!=self.minfPaddingIdx] for xs in xss]
        if self.bertMode:
            xss_wo_pad = [[[self.mlm.word2id['[PAD]'] 
                                if a==self.minfPaddingIdx else a for a in x] 
                            for x in xs] for xs in xss]
        
        ls = [len(xs) for xs in xss_wo_pad]

        xss_wo_pad_flatten = [x for xs in xss_wo_pad for x in xs]

        yss, hss = self.encoder(xss_wo_pad_flatten, self.encEmbed.weight, padding_idx=self.unkCharIdx)
        # TODO: padding_idx is confusing. change name as unk_idx
        # this proc may be related to the bilstm encoder?

        yss_pad = []
        hss_pad = []
        pointer = 0
        for l in ls:
            yss_pad.append(yss[pointer:pointer+l])
            hss_pad.append(hss[pointer:pointer+l])
            pointer += l

        yss_pad = rnn.pad_sequence(yss_pad, batch_first=True)
        
        if yss_pad.shape[1] < m:
            yss_pad = F.pad(yss_pad, (0, 0, 0, m-yss_pad.shape[1]))

        return yss_pad, hss_pad

    def __calcAttention(self, log_theta, idNbests, m, lam):

        if self.bertMode:
            idNbests = [[inb[1:-1] for inb in idNbest] for idNbest in idNbests]

        xs = [(idNth, len(idNth)) for idNbest in idNbests for idNth in idNbest]
        idNbests, lens = zip(*xs)
        maxL = max(lens)

        logPs = log_theta.unsqueeze(0)[:,[idNth + [self.zeroPaddingIdx]*(maxL-ln) for idNth, ln in zip(idNbests, lens)]] 
        logPs = torch.sum(logPs, dim=2)
        logPs = logPs.view(-1, m)

        attn = torch.exp(logPs - torch.logsumexp(logPs, dim=1, keepdim=True))

        return attn, logPs

    def __getLogTheta(self, lam, selectMode='normal'):
        # cache
        if self.training:
            self.CACHE_log_theta = None
            self.CACHE_log_smoothed_theta = None
            self.CACHE_vocab = None
        else:
            if self.CACHE_log_smoothed_theta is not None:
                return self.CACHE_log_theta, self.CACHE_log_smoothed_theta, self.CACHE_vocab

        if selectMode=='normal':
            log_theta = self.nlm.getLogUnigramProbs(self.lmEmbed)
            log_smoothed_theta = lam * log_theta
            vocab = None
        elif selectMode=='top':
            log_theta, selectedIds = self.nlm.getSelectedLogUnigramProbs(
                                                    self.lmEmbed, 
                                                    self.topK, 
                                                    mode=selectMode, 
                                                    lam=lam,
                                                    mustBeIncludeIdSet=self.mlm.getCharIdSet() | {self.unkCharIdx,})
            log_smoothed_theta = lam * log_theta
            vocab = set([self.mlm.id2word[i] for i in selectedIds])
        else:
            print('selectMode should be top or normal.'); exit()
        
        # minf padding
        log_theta = F.pad(log_theta,
                          pad=(0,1),
                          value=float('-inf'))
        log_smoothed_theta = F.pad(log_smoothed_theta,
                          pad=(0,1),
                          value=float('-inf'))
        
        # zero padding
        log_theta = F.pad(log_theta,
                          pad=(0,1),
                          value=0)
        log_smoothed_theta = F.pad(log_smoothed_theta,
                          pad=(0,1),
                          value=0)

        # cache
        if not self.training and self.CACHE_log_smoothed_theta is None:
            self.CACHE_log_theta = log_theta
            self.CACHE_log_smoothed_theta = log_smoothed_theta
            self.CACHE_vocab = vocab
        return log_theta, log_smoothed_theta, vocab

    def __getNbests(self, lines, log_theta, m, n, vocab=None):
        if vocab is None:
            vocab = self.mlm.vocab

        # nbests
        with torch.no_grad():
            log_theta = log_theta.cpu().detach().numpy().astype(float)
            idTables = [self.mlm.makeIdTable(
                            line,
                            paddingIdx=self.minfPaddingIdx,
                            unkCharIdx=self.unkCharIdx,
                            vocab=vocab
                        )  for line in lines]

            logProbTables = [self.makeLogProbTable(
                                idTable,
                                log_theta)
                             for idTable in idTables]
            
            idNbests = [mdp.mSampleFromNBestIdSegmentation(idTable, logProbTable, m, n, mode='astar')
                            for idTable, logProbTable in zip(idTables, logProbTables)]
    
            # add pad if len(idNbest) < m
            idNbests = [idNbest + ([self.minfPaddingIdx],)*(m-len(idNbest)) for idNbest in idNbests]
                
            if self.bertMode:
                CLS = [self.mlm.word2id['[CLS]']]
                SEP = [self.mlm.word2id['[SEP]']]
                idNbests = [[CLS+inb+SEP for inb in idNbest] for idNbest in idNbests]

        nbests = [[[self.mlm.id2word[i] if i in self.mlm.id2word else '[EXPAD]'
                    for i in inb] for inb in idNbest] for idNbest in idNbests]

        return nbests, idNbests

    def __getUnigramLoss(self, nbests, logPs, attn):
        if self.bertMode:
            nbests = [[nb[1:-1] for nb in nbest] for nbest in nbests]

        nonPadIdx = torch.where(logPs!=float('-inf'))
        weightedLogPs = - logPs[nonPadIdx] * attn.squeeze(1)[nonPadIdx] 
        lens_wo_pad = torch.tensor([len(nth) for nbest in nbests for nth in nbest if nth[0] != '[EXPAD]']).to(attn.device.type)

        uniLoss = torch.sum(weightedLogPs / lens_wo_pad) / weightedLogPs.shape[0] 
        return uniLoss

    def forward(self, lines):
        '''
        if you want to use different lam / selectMode for each iteration,
        change self.lam / self.selectMode directly.
        '''
        # TODO: implement setter of aboce hyperparameters
        # TODO: implement scheduler of lam

        # reset hyperparameters for inference
        n = self.n if self.training else self.nTest
        m = self.m if self.training else self.mTest
        lam = self.lam if self.training else self.lamTest
        selectMode = self.selectMode if self.training else self.selectModeTest
            
        log_theta, log_smoothed_theta, vocab = self.__getLogTheta(lam, selectMode)

        nbests, idNbests = self.__getNbests(lines, log_theta, m, n, vocab)

        # gumbel softmax
        attn, logPs = self.__calcAttention(log_smoothed_theta, idNbests, m=m, lam=lam)
        
        # encodes
        yss, hss = self.encode(idNbests, m=m)

        # weighting
        attn = attn.view(len(lines), -1, m)
        vss = torch.matmul(attn, yss).squeeze(1)

        # unigram loss
        uniLoss = self.__getUnigramLoss(nbests, logPs, attn.detach())

        return vss, uniLoss, nbests, attn, yss, hss

    def makeLogProbTable(self, idTable, theta):
        logProbTable = theta[idTable.flatten()]
        logProbTable = logProbTable.reshape(idTable.shape)
        return logProbTable

    def saveNLMasMLM(self, path):
        # make unigram dict
        theta = self.nlm.getUnigramProbs(self.lmEmbed).cpu().data.tolist()
        unigramDict = {w:theta[i] for i,w in self.mlm.id2word.items()}# if 0.<theta[i]}

        # make new mlm
        self.mlm.setVocabFromUnigramDict(unigramDict)
        self.mlm.save(path)
        print('>>> DUMP LEARNED LM AS MLM')

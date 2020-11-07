import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from time import time

class UnigramNLM(nn.Module):
    def __init__(self, vocabSize, embedSize, unkIdx):
        super().__init__()
        # vocabSize should include unk

        self.scoreModule = nn.Sequential()
        self.scoreModule.add_module('tanh1', nn.Tanh())
        self.scoreModule.add_module('linear1', nn.Linear(embedSize, int(embedSize*1.5)))
        self.scoreModule.add_module('tanh2', nn.Tanh())
        self.scoreModule.add_module('linear2', nn.Linear(int(embedSize*1.5), 1))
        
        self.vocabSize = vocabSize
        self.unkIdx = unkIdx

    def fitTo(self, embed, theta, maxEpoch=100000, thre=1e-7):
        print('>>> FIT UNIGRAM NLM TO PRETRAINED THETA')
        theta = torch.FloatTensor(theta).to(embed.weight.device.type)
        opt = torch.optim.Adam(self.parameters())

        criterion = nn.KLDivLoss(reduction='mean')

        startTime = time()
        prevLoss = -1
        for i in range(maxEpoch):
            opt.zero_grad()
            loss = criterion(self.getLogUnigramProbs(embed), theta)
            loss.backward()
            opt.step()
            print('\r%d/%d: %.10f'%((i+1), maxEpoch, loss.data.tolist()), end='')
            if loss < thre or loss==prevLoss:
                break
            prevLoss = loss
        print('\nDONE, \nLOSS=',loss)
        print('TIME:', time()-startTime)
    
    def getUnigramProbs(self, embed):
        # embed is detatched
        return F.softmax(self.scoreModule(embed.weight.detach()).flatten(), dim=0)

    def getLogUnigramProbs(self, embed):
        # embed is detatched
        return F.log_softmax(self.scoreModule(embed.weight.detach()).flatten(), dim=0)

    def getUnigramScores(self, embed):
        # embed is detatched
        return F.relu(self.scoreModule(embed.weight.detach()).flatten())

    def getSelectedLogUnigramProbs(self, embed, m, mode='sampling', lam=1.0, mustBeIncludeIdSet=None):
        if mustBeIncludeIdSet is None:
            # when given ids are None, assign unkIdx as a default.
            # if you want to set assigned Ids set as literaly empty, you should set it as set()
            mustBeIncludeIdSet = {self.unkIdx,}

        # this method returns m-1 selected items.
        # this is caused by the unk probability

        idx = set(range(self.vocabSize))
        bookedSize = len(mustBeIncludeIdSet)
        unbookedIds = list(set(range(self.vocabSize))-mustBeIncludeIdSet) # should sort?

        dist = self.getUnigramProbs(embed)
        with torch.no_grad():
            theta = dist[unbookedIds]
            theta = theta.cpu().detach().numpy()

            if mode=='sampling':
                # handle lam
                theta = theta ** lam
                theta = theta / sum(theta)
                
                selectedIdx = np.random.choice(
                                    unbookedIds,
                                    m-bookedSize, 
                                    p=theta, 
                                    replace=False)
            elif mode=='top':
                selectedIdx = np.array(unbookedIds)[np.argsort(theta)[-(m-bookedSize):]]
            else:
                print('not implimented')
                exit()
        
        if len(set(selectedIdx) & mustBeIncludeIdSet) != 0:
            print('error in selection')
            print(set(selectedIdx) & mustBeIncludeIdSet)
            exit()

        selectedIdx = set(selectedIdx) | mustBeIncludeIdSet
        unselectedIdx = list(idx - selectedIdx)

        log_dist = dist.log()
        log_dist[unselectedIdx] = float('-inf')

        return F.log_softmax(log_dist), selectedIdx


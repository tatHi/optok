from multigram import lm
import classifier
from torch import nn
import torch
import numpy as np

# assign seed
torch.manual_seed(0)
np.random.seed(0)

# load text and labels for training
text = [line.strip() for line in open('../data/train.text')]
labels = torch.LongTensor(
            [int(line) for line in open('../data/train.label')]
         )

# initialize multigram lm
mlm = lm.MultigramLM(
        maxLength=5, minFreq=4, data=text, unkToken='<unk>'
      )

# if you have pretrained sentencepiece, initialize mlm without text and load the model as following
# mlm = lm.MultigramLM()
# mlm.loadSentencePieceModel('path/to/sentencepiece.model')
#
# also, text should be preprocessed to use it in sentencepiece style:
# import unicodedata
# text = [unicodedata.normalize('NFKC', line) for line in data]
# text = ['▁'+line.replace(' ', '▁') for line in data]
      
cl = classifier.Classifier(
       mlm=mlm,
       embedSize=3,
       hidSize=2,
       labelSize=2,
       m=3,
       n=3,
       topK=4,
       lam=0.2,
       selectMode='normal',
       encoderType='bilstm',
       dropoutRate=0.5,
       useIndividualEmbed=False)

# when initializing mlm with sentencepiece, fit neural unigram lm of OpTok as:
# cl.ot.nlm.fitTo(cl.lmEmbed, mlm.theta)

lossFunc = nn.CrossEntropyLoss()
opt = torch.optim.Adam(cl.parameters())

# one step
opt.zero_grad()
ys, lmLoss = cl.forward(text)
clLoss = lossFunc(ys, labels)

print('-'*30)
print('Predicted scores')
print(ys)
print('-'*30)
print('Classification loss')
print(clLoss)
print('-'*30)
print('Language model loss')
print(lmLoss)
print('-'*30)

clLoss.backward()
opt.step()


#   -----------   #
#   SAVE MODELS   #
#   -----------   #

# save trained model
torch.save(cl.state_dict(), 'test_dir/cl.model')

# save trained lm for tokenizer
cl.ot.saveNLMasMLM('test_dir/mlm.model')


#   ---------------------------------   #
#   LOAD AND TOKENIZE WITH TRAINED LM   #
#   ---------------------------------   #
from multigram import tokenizer

# load trained mlm
mlm = lm.MultigramLM()
mlm.load('test_dir/mlm.model')

# set mlm to tokenizer
tknzr = tokenizer.Tokenizer(mlm)

# tokenize
print('Tokenization')
print('-'*30)
for line in text:
    print('pieces:', tknzr.encode_as_pieces(line))
    print('ids   :', tknzr.encode_as_ids(line))
    print('-'*30)

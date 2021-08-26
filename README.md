# Optimizing Word Segmentation for Downstream Task
Author's implementation of "[Optimizing Word Segmentation for Downstream Task](https://www.aclweb.org/anthology/2020.findings-emnlp.120.pdf)".
In short, we call our system "OpTok: Optimizing Tokenization".

FYI: We extended OpTok to be used for various NLP tasks in "[Joint Optimization of Tokenization and Downstream Model](https://arxiv.org/abs/2105.12410)" and you can access [the official implementation](https://github.com/tatHi/optok4at).

## Requirements
- [multigram v0.1](https://github.com/tatHi/multigram/tree/v0.1)
- numpy==1.18.0
- torch==1.6.0+cu101
- (transformers==2.8.0, if you use BERT as an encoder)

## Setup
Install [multigram v0.1](https://github.com/tatHi/multigram/tree/v0.1) and prepare OpTok repository.

```
$ mkdir optok_environment
$ cd optok_environment
$ git clone https://github.com/tatHi/multigram -b v0.1
$ git clone https://github.com/tatHi/optok
$ cd multigram
$ pip install --editable .
```

## Run exapmle
`/src/run_example.py` describes example codes of training OpTok, dumping models, and tokenize text with a trained language model.

```
$ cd optok/src
$ mkdir test_dir
$ python run_example.py
>>> BUILD VOCABULARY
possible n-grams (n=5): 9
>>> INITIALIZE THETA
------------------------------
Predicted scores
tensor([[0.6488, 0.8476],
        [0.6342, 0.7852],
        [0.5614, 0.4838]], grad_fn=<AddmmBackward>)
------------------------------
Classification loss
tensor(0.7169, grad_fn=<NllLossBackward>)
------------------------------
Language model loss
tensor(0.1875, grad_fn=<DivBackward0>)
------------------------------
>>> DUMP LEARNED LM AS MLM
Tokenization
------------------------------
pieces: ['a', 'b', 'cd', 'e', 'f', 'g']
ids   : [1, 2, 4, 6, 7, 8]
------------------------------
pieces: ['cd', 'a', 'b', 'c', 'cd']
ids   : [4, 1, 2, 3, 4]
------------------------------
pieces: ['a', 'b', 'b', 'cd', 'e']
ids   : [1, 2, 2, 4, 6]
------------------------------
```

# Experimental Settings
Training split of Amazon Dataset and Twitter(Ja) dataset used in the paper is available [here](https://drive.google.com/drive/folders/1MQRrKPFE53i8qP5m8gyypCl4cwV5Vkv8?usp=sharing).
The google drive also includes pre-trained word embeddings and SentencePiece model for each experiment.

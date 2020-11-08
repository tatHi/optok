# Optimizing Word Segmentation for Downstream Task
Author's implementation of "Optimizing Word Segmentation for Downstream Task".
In short, we call our system "OpTok: Optimizing Tokenization".

## Requirements
- [multigram](https://github.com/tatHi/multigram)
- transformers==2.8.0
- numpy==1.18.0
- torch==1.6.0+cu101

## Setup
Install [multigram](https://github.com/tatHi/multigram) and prepare OpTok repository.

```
$ mkdir optok_environment
$ cd optok_environment
$ git clone https://github.com/tatHi/multigram
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

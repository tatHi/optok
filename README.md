# Optimizing Word Segmentation for Downstream Task
Author's implementation of "Optimizing Word Segmentation for Downstream Task".
In short, we call our system "OpTok: Optimizing Tokenization".

## Requirements
- [multigram](https://github.com/tatHi/multigram)
- PyTorch
- numba

## Setup
```
$ mkdir optok_environment
$ cd optok_environment
$ git clone https://github.com/tatHi/multigram
$ git clone https://github.com/tatHi/optok
$ cd multigram
$ pip install --editable .
```

## Run exapmle
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
```

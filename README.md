# PBLM-Domain-Adaptation
Authors: Yftah Ziser, Roi Reichart (Technion - Israel Institute of Technology).

This is a code repository used to generate the results appearing in [Pivot Based Language Modeling for Improved Neural Domain Adaptation](http://www.aclweb.org/anthology/N18-1112).

If you use this implementation in your article, please cite :)
```bib
@inproceedings{ziser2018pivot,
  title={Pivot Based Language Modeling for Improved Neural Domain Adaptation},
  author={Ziser, Yftah and Reichart, Roi},
  booktitle={Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)},
  volume={1},
  pages={1241--1251},
  year={2018}
}
```

## INSTALLATION

PBLM requires the following packages:

Python >= 2.7.

numpy

scipy

Theano

keras

scikit-learn

## Example
You can find an explained example in run.py: 

```python
import tr
import sentiment
import pre
import os
import itertools

if __name__ == '__main__':
    domain = []
    domain.append("books")
    domain.append("kitchen")
    domain.append("dvd")
    domain.append("electronics")

    # training the PBLM model in order to create structure aware representation for domain adaptation
    #input:
    # shared representation for both source domain and target domain
    # first param: the source domain
    # second param: the target domain
    # third param: number of pivots
    # fourth param: appearance threshold for pivots in source and target domain
    # fifth param: the embedding dimension
    # sixth param: maximum number of words to work with
    # seventh param: maximum review length to work with
    # eighth param: hidden units number for the PBLM model
    #output: the software will create corresponding directory with the model
    tr.train_PBLM(domain[0], domain[1], 500, 10, 256, 10000, 500, 256)

    # training the sentiment cnn using PBLM's representation
    # shared representation for both source domain and target domain
    # this phase needs a corresponding trained PBLM model in order to work
    # first param: the source domain
    # second param: the target domain
    # third param: number of pivots
    # fourth param: maximum review length to work with
    # fifth param: the embedding dimension
    # sixth param: maximum number of words to work with
    # seventh param: hidden units number for the PBLM model
    # eighth param: the number of filters for the CNN
    # ninth param: the kernel size for the CNN
    # output: the results file will be created in the same directory
    # of the model under the results directory in the "cnn" dir
    sentiment.PBLM_CNN(domain[0], domain[1], 500, 500, 256, 10000, 256, 250, 3)

    # training the sentiment cnn using PBLM's representation
    # shared representation for both source domain and target domain
    # this phase needs a corresponding trained PBLM model in order to work
    # first param: the source domain
    # second param: the target domain
    # third param: number of pivots
    # fourth param: maximum review length to work with
    # fifth param: the embedding dimension
    # sixth param: maximum number of words to work with
    # seventh param: hidden units number for the PBLM model
    # eighth param: hidden units number for the lstm model
    # output: the results file will be created in the same directory
    # of the model under the results directory in the "lstm" dir
    sentiment.PBLM_LSTM(domain[0], domain[1], 500, 500, 256, 10000, 256, 256)
```

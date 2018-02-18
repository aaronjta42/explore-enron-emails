#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Script that inferences doc2vec model
"""

from gensim.models import Doc2Vec

from utils import pprint


if __name__ == "__main__":
    model = Doc2Vec.load('./model.d2v')

    pprint("Most similar to 'good': {}".format(model.most_similar('good')))

    example_sentence = model['TRAIN_ALLEN_P_0']
    pprint("TRAIN_ALLEN_P_0 email: {}".format(example_sentence))
    pprint("# Dimensions in sentence: {}".format(len(example_sentence)))

    pprint(model.corpus_count)
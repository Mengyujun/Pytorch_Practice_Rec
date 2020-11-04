from collections import defaultdict, namedtuple, OrderedDict

import torch
import torch.nn as nn
import numpy as np

DEFAULT_GROUP_NAME = "default_group"

class SparseFeat(namedtuple("SparseFeat", ['name', 'vocabulary_size', 'embedding_dim', 'dtype',
                                           'embedding_name', 'group_name'])):
    __slots__ = ()

    def __new__(cls, name, vocabulary_size, embedding_dim=4, dtype="int32", embedding_name=None,
                ground_name=DEFAULT_GROUP_NAME):
        if embedding_name = None:
            embedding_name = name
        if embedding_dim = 'auto':
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        return  super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, dtype, embedding_name,
                                               ground_name)


class DenseFeat(namedtuple("DenseFeat", ['name', 'dimension', 'dtype'])):
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype="float32"):
        return super(DenseFeat, self).__new__(cls, name, dimension, dtype)


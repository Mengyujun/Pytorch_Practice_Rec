import torch
import torch.nn as nn
import numpy as np

from collections import defaultdict, namedtuple, OrderedDict
from .layers.utils import concat_fun

DEFAULT_GROUP_NAME = "default_group"

class SparseFeat(namedtuple("SparseFeat", ['name', 'dimension', 'use_hash', 'dtype','embedding_name','embedding'])):
    __slots__ = ()

    def __new__(cls, name, dimension, use_hash=False, dtype="int32", embedding_name=None, embedding=True):
        if embedding_name == None:
            embedding_name = name
        # if embedding_dim == 'auto':
        #     embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        return  super(SparseFeat, cls).__new__(cls, name, dimension, use_hash, dtype, embedding_name, embedding)


class DenseFeat(namedtuple("DenseFeat", ['name', 'dimension', 'dtype'])):
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype="float32"):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)



def combined_dnn_input(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = torch.cat(sparse_embedding_list, dim=-1).squeeze()
        dense_dnn_input = torch.cat(dense_value_list, dim=-1).squeeze()

        return concat_fun([sparse_dnn_input, dense_dnn_input])

    elif len(sparse_embedding_list) > 0:
        return torch.cat(sparse_embedding_list, dim=-1).squeeze()
    elif len(dense_value_list) > 0 :
        return torch.cat(dense_value_list, dim=-1).squeeze()
    else:
        raise NotImplementedError

def get_feature_names(feature_columns):
    feature_dict = build_input_features(feature_columns)
    return list(feature_dict.keys())


def build_input_features(feature_columns):
    features = OrderedDict()

    start = 0

    for feat in feature_columns:
        feat_name = feat.name
        if feat_name in features:
            continue
        if isinstance(feat, SparseFeat):
            features[feat_name] = (start, start+1)
            start += 1
        elif isinstance(feat, DenseFeat):
            features[feat_name] = (start, start+feat.dimension)
            start += feat.dimension

    return features



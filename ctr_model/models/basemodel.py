import torch
import torch.nn as nn

from ctr_model.inputs import SparseFeat, DenseFeat
from ctr_model.layers import PredictionLayer


class Linear(nn.Module):
    def __init__(self, feature_columns, feature_index, init_std=0.0001, device='cpu'):
        super(Linear, self).__init__()

        self.feature_index = feature_index
        self.sparse_feature_columns = list(
            filter(lambda x:isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        self.embedding_dict = self.create_embedding_matrix(self.sparse_feature_columns, 1, init_std,
                                                           sparse=False).to(device)

        for tensor in self.embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)
        
        if len(self.dense_feature_columns) > 0:
            self.weight = nn.Parameter(torch.tensor(len(self.dense_feature_columns), 1))
            nn.init.normal(self.weight, mean=0, std=init_std)


    def forward(self, X):
        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long())
            for feat in self.sparse_feature_columns]

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]]
                              for feat in self.dense_feature_columns]

        if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
            linear_sparse_logit = torch.sum(torch.cat(sparse_embedding_list, dim=-1), dim=-1, keepdim=False)
            linear_dense_logit = torch.cat(dense_value_list, dim=-1).matmul(self.weight)
            linear_logit = linear_sparse_logit + linear_dense_logit
        elif len(sparse_embedding_list) > 0:
            linear_logit = torch.sum(torch.cat(sparse_embedding_list, dim=-1), dim=-1, keepdim=False)
        elif len(dense_value_list) > 0:
            linear_logit = torch.cat(dense_value_list, dim=-1).matmul(self.weight)
        else:
            raise NotImplementedError
        return linear_logit

    def create_embedding_matrix(self, feature_columns, embedding_size, init_std=0.0001, sparse=False):
        # return  nn.ModuleDict: for sparse features, {embedding_name: nn.Embedding}
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []

        embedding_dict = nn.ModuleDict(
            {feat.embedding_name: nn.Embedding(feat.dimension, embedding_size, sparse=sparse) for feat in
            sparse_feature_columns})

        for tensor in embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)

        return embedding_dict


class BaseModel(nn.Module):
    def __init__(self, 
                 linear_feature_columns, dnn_feature_columns, embedding_size=8, dnn_hidden_units=(128, 128), 
                 l2_reg_linear=1e-5,
                 l2_reg_embedding=1e-5, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu',
                 task='binary', device='cpu'):
        super(BaseModel, self).__init__()
        
        self.reg_loss = torch.zeros((1,), device=device)
        self.device = device 
        
        self.feature_index = build_input_features(linear_feature_columns + dnn_feature_columns)
        self.dnn_feature_columns = dnn_feature_columns

        self.embedding_dict = self.create_embedding_matrix(dnn_feature_columns, embedding_size, init_std,
                                                           sparse=False).to(device)

        self.linear_model = Linear(linear_feature_columns, self.feature_index, device=device)

        # add regularization loss
        self.add_regularization_loss(self.embedding_dict.parameters(), l2_reg_embedding)
        self.add_regularization_loss(self.linear_model.parameters(), l2_reg_linear)

        self.out = PredictionLayer(task, )
        self.to(device)

    def fit(self):
        pass

    def evaluate(self):
        pass

    def predict(self):
        pass

    def input_from_feature_columns(self, X, feature_columns, embedding_dict):
        sparse_feature_columns = list(
            filter(lambda x:isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        sparse_embedding_list = [embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long())
            for feat in sparse_feature_columns]

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]]
                            for feat in dense_feature_columns]

        return sparse_embedding_list, dense_value_list

    def add_regularization_loss(self, weight_list, weight_deecay, p=2):
        reg_loss = torch.zeros((1,), device=device)
        for w in weight_list:
            l2_reg = torch.norm(w, p=p, )
            reg_loss += l2_reg
        reg_loss = weight_deecay * reg_loss
        self.reg_loss += reg_loss

    def create_embedding_matrix(self, feature_columns, embedding_size, init_std=0.0001, sparse=False):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []

        embedding_dict = nn.ModuleDict(
            {feat.embedding_name: nn.Embedding(feat.dimension, embedding_size, sparse=sparse) for feat in
             sparse_feature_columns}
        )
        for tensor in embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)


    def compute_input_dim(self, feature_columns, embedding_size, dense_only=False):
        sparse_feature_columns = list(
            filter(lambda x:isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        if dense_only:
            return sum(map(lambda x:x.dimension, dense_feature_columns))
        else :
            return len(sparse_feature_columns) * embedding_size + sum(map(lambda x:x.dimension, dense_feature_columns))
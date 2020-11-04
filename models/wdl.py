import torch
import torch.nn as nn
import torch.nn.functional as F

from .basemodel import BaseModel
from ..layers import DNN

class WDL(BaseModel):
    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, embedding_size=8, dnn_hidden_units=(128, 128),
                 l2_reg_linear=1e-5,
                 l2_reg_embedding=1e-5, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu',
                 task='binary', device='cpu'):
        super(WDL, self).__init__(linear_feature_columns, dnn_feature_columns, embedding_size=embedding_size,
                                  dnn_hidden_units=dnn_hidden_units,
                                  l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding,
                                  l2_reg_dnn=l2_reg_dnn, init_std=init_std, seed=seed, dnn_dropout=dnn_dropout,
                                  dnn_activation=dnn_activation, task=task, device=device)

        self.dnn = DNN(self.compute_input_dim(dnn_feature_columns, embedding_size, ), dnn_hidden_units,
                       activation=dnn_activation, l2_reg=l2_reg_dnn,dropout_rate=dnn_dropout, init_std=init_std)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
        self.add_regularization_loss(chain(self.dnn.parameters(), self.dnn_linear.parameters()), l2_reg_dnn)
        self.to(device)

    def forward(self, X):
        pass
# =========================================================================
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import EmbeddingLayer, MLP_Layer, PoolLayer


class DNN_pool(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="DNN",
                 gpu=-1,
                 task="binary_classification",
                 learning_rate=1e-3,
                 embedding_initializer="torch.nn.init.normal_(std=1e-4)",
                 embedding_dim=10,
                 hidden_units=[64, 64, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 num_cluster=10,
                 pool_mlp_layers=2,
                 pool_attention_layers=2,
                 **kwargs):
        super(DNN_pool, self).__init__(feature_map,
                                  model_id=model_id,
                                  gpu=gpu,
                                  embedding_regularizer=embedding_regularizer,
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        self.pool = PoolLayer(feature_map.num_fields, num_clusters=num_cluster, embedding_dim=embedding_dim, mlp_layers=pool_mlp_layers, net_dropout=net_dropout,
                              pool_attention_layers=pool_attention_layers)
        self.dnn = MLP_Layer(input_dim=embedding_dim * (feature_map.num_fields + num_cluster),
                             output_dim=1,
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=self.get_output_activation(task),
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)
        cluster_emb = self.pool(feature_emb)
        all_emb = torch.cat([feature_emb, cluster_emb], dim=1)
        y_pred = self.dnn(all_emb.flatten(start_dim=1))
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict
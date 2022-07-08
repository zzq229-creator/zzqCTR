import torch
from torch import nn
from fuxictr.pytorch.layers import MLP_Layer, MultiHeadSelfAttention


class PoolLayer(nn.Module):
    def __init__(self, num_fields, num_clusters, embedding_dim, mlp_layers, net_dropout, pool_attention_layers, softmax_dim=-1):
        super(PoolLayer, self).__init__()
        hidden_dim = embedding_dim * 2
        self.self_attention = nn.Sequential(
            *[MultiHeadSelfAttention(embedding_dim if i == 0 else hidden_dim,
                                     attention_dim=hidden_dim,
                                     num_heads=1,
                                     dropout_rate=net_dropout,
                                     use_residual=False,
                                     align_to="output")
              for i in range(pool_attention_layers)])
        self.dense_S = MLP_Layer(input_dim=hidden_dim * 2,
                                 output_dim=num_clusters,
                                 hidden_units=[hidden_dim * 5] * mlp_layers,
                                 output_activation=None,
                                 dropout_rates=net_dropout)
        self.softmax = nn.Softmax(dim=softmax_dim)

    def forward(self, input_tensor):
        hidden = self.self_attention(input_tensor)
        hidden_sum = torch.sum(hidden, dim=1, keepdim=True)
        hidden_sum = torch.tile(hidden_sum, [1, hidden.shape[1], 1])
        S = self.dense_S(torch.concat([hidden, hidden_sum], -1))
        S = self.softmax(S)
        S = torch.transpose(S, 1, 2)
        res = torch.matmul(S, input_tensor)
        return res

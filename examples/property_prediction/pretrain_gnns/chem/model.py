import torch.nn as nn
from dgl.nn.pytorch.glob import GlobalAttentionPooling, SumPooling, AvgPooling, MaxPooling
from dgllife.model.gnn.gin import GIN


class GINMaskingModel(nn.Module):
    """
    GIN model adapted from dgllife.model.gnn.gin.GIN
    Used for the pretrain_masking task.
    """
    def __init__(self, num_node_emb_list, num_edge_emb_list, num_layers=5,
                 emb_dim=300, JK='last', dropout=0.5, readout='mean', n_tasks=300):
        super(GINMaskingModel, self).__init__()

        if num_layers < 2:
            raise ValueError('Number of GNN layers must be greater '
                             'than 1, got {:d}'.format(num_layers))

        self.gnn = GIN(num_node_emb_list=num_node_emb_list,
                       num_edge_emb_list=num_edge_emb_list,
                       num_layers=num_layers,
                       emb_dim=emb_dim,
                       JK=JK,
                       dropout=dropout)

        if JK == 'concat':
            self.predict = nn.Linear((num_layers + 1) * emb_dim, n_tasks)
        else:
            self.predict = nn.Linear(emb_dim, n_tasks)

    def forward(self, g, categorical_node_feats, categorical_edge_feats):
        node_feats = self.gnn(g, categorical_node_feats, categorical_edge_feats)
        return self.predict(node_feats)

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv, ChebConv, SAGEConv, GATConv, TAGConv


# GCN model
class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_dim, hidden_dim))
        for _ in range(1, num_layers):
            self.layers.append(GraphConv(hidden_dim, hidden_dim))
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, blocks, features):
        for layer, block in zip(self.layers, blocks):
            features = layer(block, features).flatten(1)
        output = self.fc(features)
        return output

# ChebyNet model
class ChebyNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, k, num_layers=2):
        super(ChebyNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(ChebConv(in_dim, hidden_dim, k))
        for _ in range(1, num_layers):
            self.layers.append(ChebConv(hidden_dim, hidden_dim, k))
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, blocks, features):
        for layer, block in zip(self.layers, blocks):
            features = layer(block, features).flatten(1)
        output = self.fc(features)
        return output


# GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_dim, hidden_dim, aggregator_type='mean'))
        for _ in range(1, num_layers):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim, aggregator_type='mean'))
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, blocks, features):
        for layer, block in zip(self.layers, blocks):
            features = layer(block, features).flatten(1)
        output = self.fc(features)
        return output

# TAGCN model
class TAGGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, k , num_layers=2):
        super(TAGGCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(TAGConv(in_dim, hidden_dim, k))
        for _ in range(1, num_layers):
            self.layers.append(TAGConv(hidden_dim, hidden_dim, k))
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, blocks, features):
        for layer, block in zip(self.layers, blocks):
            features = layer(block, features).flatten(1)
        output = self.fc(features)
        return output

# GAT model
class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, num_layers=2):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_dim, hidden_dim, num_heads))
        for _ in range(1, num_layers):
            self.layers.append(GATConv(hidden_dim * num_heads, hidden_dim, num_heads))
        self.fc = nn.Linear(hidden_dim * num_heads, out_dim)

    def forward(self, blocks, features):
        for layer, block in zip(self.layers, blocks):
            features = layer(block, features).flatten(1)
        output = self.fc(features)
        return output






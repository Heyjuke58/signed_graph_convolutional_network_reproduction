from torch_geometric.nn import SignedGCN
from build_graphs import build_edge_indices, MY_GRAPH

in_features = 64
out_features = 64
num_layers = 2
lamb = 5

sgcn = SignedGCN(in_features, out_features, num_layers, lamb)

pos_edge_index, neg_edge_index, num_nodes = build_edge_indices(*MY_GRAPH)
bla = sgcn.create_spectral_features(pos_edge_index, neg_edge_index, num_nodes)
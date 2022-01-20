from torch import LongTensor
from torch_geometric.nn import SignedGCN


class SSETrainer:
    def __init__(
        self,
        train_pos_edge_index: LongTensor,
        train_neg_edge_index: LongTensor,
        test_pos_edge_index: LongTensor,
        test_neg_edge_index: LongTensor,
        num_nodes: int,
        embedding_size: int,
    ) -> None:
        """
        The edge indices should all be undirected???
        """
        self.train_pos_edge_index = train_pos_edge_index
        self.train_neg_edge_index = train_neg_edge_index
        self.test_pos_edge_index = test_pos_edge_index
        self.test_neg_edge_index = test_neg_edge_index
        self.num_nodes = num_nodes

        # we abuse this class for its implementation of SSE
        self.sgcn = SignedGCN(embedding_size * 2, 1, 1)
        self.embedding_size = embedding_size

        self.X = self.sgcn.create_spectral_features(
            self.train_pos_edge_index, self.train_neg_edge_index, self.num_nodes
        )

    def train(self):
        return self.X[:, : self.embedding_size]

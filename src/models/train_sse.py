from torch import LongTensor
from torch_geometric.nn import SignedGCN
from src.models import Trainer
import torch

DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class SSETrainer(Trainer):
    def __init__(
        self,
        train_pos_edge_index: LongTensor,
        train_neg_edge_index: LongTensor,
        val_pos_edge_index: LongTensor,
        val_neg_edge_index: LongTensor,
        test_pos_edge_index: LongTensor,
        test_neg_edge_index: LongTensor,
        num_nodes: int,
        embedding_size_true: int,
        embedding_size_used: int,
    ) -> None:
        """
        The edge indices should all be undirected???
        """
        super().__init__(
            train_pos_edge_index,
            train_neg_edge_index,
            val_pos_edge_index,
            val_neg_edge_index,
            test_pos_edge_index,
            test_neg_edge_index,
            num_nodes,
        )
        self.embedding_size_true = embedding_size_true
        self.embedding_size_used = embedding_size_used

        # we abuse this class for its implementation of SSE
        self.sgcn = SignedGCN(embedding_size_true, 1, 1).to(DEV)

        self.X = self.sgcn.create_spectral_features(
            self.train_pos_edge_index, self.train_neg_edge_index, self.num_nodes
        ).to(DEV)

    def train(self, plot: bool):
        return self.X[:, : self.embedding_size_used]

    def get_num_parameters(self) -> int:
        return 0

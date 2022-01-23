from abc import ABC, abstractmethod
from torch import Tensor, LongTensor


class Trainer(ABC):
    def __init__(
        self,
        train_pos_edge_index: LongTensor,
        train_neg_edge_index: LongTensor,
        val_pos_edge_index: LongTensor,
        val_neg_edge_index: LongTensor,
        test_pos_edge_index: LongTensor,
        test_neg_edge_index: LongTensor,
        num_nodes: int,
    ):
        self.train_pos_edge_index = train_pos_edge_index
        self.train_neg_edge_index = train_neg_edge_index
        self.val_pos_edge_index = val_pos_edge_index
        self.val_neg_edge_index = val_neg_edge_index
        self.test_pos_edge_index = test_pos_edge_index
        self.test_neg_edge_index = test_neg_edge_index
        self.num_nodes = num_nodes

    @abstractmethod
    def train(self, verbose: bool, plot: bool) -> Tensor:
        """
        train the model, return the learned embedding
        """
        raise NotImplementedError

    def get_num_parameters(self) -> int:
        raise NotImplementedError

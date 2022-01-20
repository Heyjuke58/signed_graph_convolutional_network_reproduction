from torch_geometric.nn import SignedGCN
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling


class OurSignedGCN(SignedGCN):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers,
        lamb=5,
        bias=True,
        xent_weights=[0.15, 0.8, 0.05],
    ):
        super().__init__(in_channels, hidden_channels, num_layers, lamb, bias)
        self.xent_weights = xent_weights

    def nll_loss(self, z, pos_edge_index, neg_edge_index):
        """Computes the discriminator loss based on node embeddings :obj:`z`,
        and positive edges :obj:`pos_edge_index` and negative nedges
        :obj:`neg_edge_index`.

        Args:
            z (Tensor): The node embeddings.
            pos_edge_index (LongTensor): The positive edge indices.
            neg_edge_index (LongTensor): The negative edge indices.
            weights (List): Three weights. For: positive edges, negative edges, not-edges
        """

        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        none_edge_index = negative_sampling(edge_index, z.size(0))

        nll_loss = 0
        nll_loss += (
            F.nll_loss(
                self.discriminate(z, pos_edge_index),
                pos_edge_index.new_full((pos_edge_index.size(1),), 0),
            )
            * self.xent_weights[0]
        )
        nll_loss += (
            F.nll_loss(
                self.discriminate(z, neg_edge_index),
                neg_edge_index.new_full((neg_edge_index.size(1),), 1),
            )
            * self.xent_weights[1]
        )
        nll_loss += (
            F.nll_loss(
                self.discriminate(z, none_edge_index),
                none_edge_index.new_full((none_edge_index.size(1),), 2),
            )
            * self.xent_weights[2]
        )
        return nll_loss / sum(self.xent_weights)

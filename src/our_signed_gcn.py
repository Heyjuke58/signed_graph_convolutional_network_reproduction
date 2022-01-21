from torch_geometric.nn import SignedGCN, SignedConv
import torch.nn.functional as F
import torch
from torch_geometric.utils import negative_sampling
from typing import Callable, Optional


class OurSignedGCN(SignedGCN):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers,
        lamb=5,
        bias=True,
        xent_weights=[0.15, 0.8, 0.05],
        activation_fn: Callable = torch.relu,
        ablation_version: str = "sgcn2",  # can be 'sgcn1', 'sgcn1p' or, 'sgcn2'
    ):
        super().__init__(in_channels, hidden_channels, num_layers, lamb, bias)
        assert ablation_version in ["sgcn2", "sgcn1", "sgcn1p"]
        self.xent_weights = xent_weights
        self.activation_fn = activation_fn
        if ablation_version == "sgcn1":
            assert num_layers == 1, "SGCN-1 must have exactly 1 layer"
            self.convs = torch.nn.ModuleList()
        elif ablation_version == "sgcn1p":
            assert num_layers == 2, "SGCN-1+ must have exactly 2 layers"
            self.convs = torch.nn.ModuleList()
            self.convs.append(SignedConv(in_channels, hidden_channels // 2, first_aggr=True))

    def forward(self, x, pos_edge_index, neg_edge_index):
        """Computes node embeddings :obj:`z` based on positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`.

        Args:
            x (Tensor): The input node features.
            pos_edge_index (LongTensor): The positive edge indices.
            neg_edge_index (LongTensor): The negative edge indices.
        """
        z = self.activation_fn(self.conv1(x, pos_edge_index, neg_edge_index))
        for conv in self.convs:
            z = self.activation_fn(conv(z, pos_edge_index, neg_edge_index))
        return z

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

    # def pos_embedding_loss(self, z, pos_edge_index):
    #     """Computes the triplet loss between positive node pairs and sampled
    #     non-node pairs.
    #
    #     Args:
    #         z (Tensor): The node embeddings.
    #         pos_edge_index (LongTensor): The positive edge indices.
    #     """
    #     i, j, k = structured_negative_sampling(pos_edge_index, z.size(0))
    #
    #     out = (z[i] - z[j]).pow(2).sum(dim=1) - (z[i] - z[k]).pow(2).sum(dim=1)
    #     return torch.clamp(out, min=0).mean()
    #
    # def neg_embedding_loss(self, z, neg_edge_index):
    #     """Computes the triplet loss between negative node pairs and sampled
    #     non-node pairs.
    #
    #     Args:
    #         z (Tensor): The node embeddings.
    #         neg_edge_index (LongTensor): The negative edge indices.
    #     """
    #     i, j, k = structured_negative_sampling(neg_edge_index, z.size(0))
    #
    #     out = (z[i] - z[k]).pow(2).sum(dim=1) - (z[i] - z[j]).pow(2).sum(dim=1)
    #     return torch.clamp(out, min=0).mean()
    #
    # def loss(self, z, pos_edge_index, neg_edge_index):
    #     """Computes the overall objective.
    #
    #     Args:
    #         z (Tensor): The node embeddings.
    #         pos_edge_index (LongTensor): The positive edge indices.
    #         neg_edge_index (LongTensor): The negative edge indices.
    #     """
    #     nll_loss = self.nll_loss(z, pos_edge_index, neg_edge_index)
    #     loss_1 = self.pos_embedding_loss(z, pos_edge_index)
    #     loss_2 = self.neg_embedding_loss(z, neg_edge_index)
    #     return nll_loss + self.lamb * (loss_1 + loss_2)
    #
    #

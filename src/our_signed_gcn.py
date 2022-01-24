from torch_geometric.nn import SignedGCN, SignedConv
import torch.nn.functional as F
import torch
from torch_geometric.utils import negative_sampling
from typing import Callable, Optional, Dict, List, Set
from random import randint
from torch import Tensor

DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.xent_weights, device=DEV))
        self.activation_fn = activation_fn
        self.distance_fn = torch.nn.PairwiseDistance(p=2)
        if ablation_version == "sgcn1":
            assert num_layers == 1, "SGCN-1 must have exactly 1 layer"
            self.convs = torch.nn.ModuleList()
        elif ablation_version == "sgcn1p":
            assert num_layers == 2, "SGCN-1+ must have exactly 2 layers"
            self.convs = torch.nn.ModuleList()
            self.convs.append(SignedConv(hidden_channels, hidden_channels // 2, first_aggr=True))

        self.conv1.to(DEV)
        self.convs.to(DEV)
        self.lin.to(DEV)

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

    # adapted from the original code of the paper
    def their_loss(
        self,
        center_nodes: List[int],
        adj_lists_pos: Dict[int, Set],
        adj_lists_neg: Dict[int, Set],
        num_nodes: int,
        embedding: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
    ):
        max_node_index = num_nodes - 1
        # get the correct nodes based on this minibatch
        i_loss2 = []
        pos_no_loss2 = []
        no_neg_loss2 = []

        i_indices = []
        j_indices = []
        ys = []
        all_nodes_set = set()
        skipped_nodes = []
        for i in center_nodes:
            # if no links then we can ignore
            if (len(adj_lists_pos[i]) + len(adj_lists_neg[i])) == 0:
                skipped_nodes.append(i)
                continue
            all_nodes_set.add(i)
            for j_pos in adj_lists_pos[i]:
                i_loss2.append(i)
                pos_no_loss2.append(j_pos)
                while True:
                    temp = randint(0, max_node_index)
                    if (temp not in adj_lists_pos[i]) and (temp not in adj_lists_neg[i]):
                        break
                no_neg_loss2.append(temp)
                all_nodes_set.add(temp)

                i_indices.append(i)
                j_indices.append(j_pos)
                ys.append(0)
                all_nodes_set.add(j_pos)
            for j_neg in adj_lists_neg[i]:
                i_loss2.append(i)
                no_neg_loss2.append(j_neg)
                while True:
                    temp = randint(0, max_node_index)
                    if (temp not in adj_lists_pos[i]) and (temp not in adj_lists_neg[i]):
                        break
                pos_no_loss2.append(temp)
                all_nodes_set.add(temp)

                i_indices.append(i)
                j_indices.append(j_neg)
                ys.append(1)
                all_nodes_set.add(j_neg)

            need_samples = 2  # number of sampling of the no links pairs
            cur_samples = 0
            while cur_samples < need_samples:
                temp_samp = randint(0, max_node_index)
                if (temp_samp not in adj_lists_pos[i]) and (temp_samp not in adj_lists_neg[i]):
                    # got one we can use
                    i_indices.append(i)
                    j_indices.append(temp_samp)
                    ys.append(2)
                    all_nodes_set.add(temp_samp)
                cur_samples += 1

        all_nodes_map = {}
        all_nodes_list = list(all_nodes_set)
        all_nodes_map = {node: i for i, node in enumerate(all_nodes_list)}

        final_embedding = self.forward(embedding, pos_edge_index, neg_edge_index)
        final_embedding = final_embedding[all_nodes_list]

        i_indices_mapped = [all_nodes_map[i] for i in i_indices]
        j_indices_mapped = [all_nodes_map[j] for j in j_indices]
        ys = torch.LongTensor(ys).to(DEV)

        # now that we have the mapped indices and final embeddings we can get the loss
        avg_loss = self.loss_fn(
            self.lin(
                torch.cat(
                    (final_embedding[i_indices_mapped], final_embedding[j_indices_mapped]), 1
                ),
            ),
            ys,
        )

        i_loss2 = [all_nodes_map[i] for i in i_loss2]
        pos_no_loss2 = [all_nodes_map[i] for i in pos_no_loss2]
        no_neg_loss2 = [all_nodes_map[i] for i in no_neg_loss2]

        avg_loss2 = torch.mean(
            torch.max(
                torch.zeros(len(i_loss2)).to(DEV),
                self.distance_fn(final_embedding[i_loss2], final_embedding[pos_no_loss2]) ** 2
                - self.distance_fn(final_embedding[i_loss2], final_embedding[no_neg_loss2]) ** 2,
            )
        )

        return avg_loss + self.lamb * avg_loss2

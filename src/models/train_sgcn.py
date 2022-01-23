from torch_geometric.nn import SignedGCN
from collections import defaultdict
import copy
import torch
from torch import Tensor, LongTensor
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Callable, Dict
from torch.optim import Adam
from src.our_signed_gcn import OurSignedGCN
from src.models import Trainer
import numpy as np


class SGCNTrainer(Trainer):
    def __init__(
        self,
        train_pos_edge_index: LongTensor,
        train_neg_edge_index: LongTensor,
        val_pos_edge_index: LongTensor,
        val_neg_edge_index: LongTensor,
        test_pos_edge_index: LongTensor,
        test_neg_edge_index: LongTensor,
        num_nodes: int,
        in_features: int,
        out_features: int,
        num_layers: int,
        lamb: float,
        num_epochs: Optional[int],
        num_batches: Optional[int],
        loss_version: str,
        learning_rate: float = 0.01,
        weight_decay: float = 1e-5,
        learn_decay: float = 0.75,
        xent_weights: List[float] = [0.15, 0.8, 0.05],
        activation_fn: Callable = torch.relu,
        ablation_version: str = "sgcn2",
        batch_size: Optional[int] = None,
        val_interval: int = 5,
        early_stopping_patience: int = 5,
    ) -> None:
        super().__init__(
            train_pos_edge_index,
            train_neg_edge_index,
            val_pos_edge_index,
            val_neg_edge_index,
            test_pos_edge_index,
            test_neg_edge_index,
            num_nodes,
        )

        self.sgcn = OurSignedGCN(
            in_features,
            out_features,
            num_layers,
            lamb,
            xent_weights=xent_weights,
            activation_fn=activation_fn,
            ablation_version=ablation_version,
        )

        self.val_interval = val_interval
        self.early_stopping_patience = early_stopping_patience

        # their loss version goes over mini-batches and uses SGD with a scheduler:
        if loss_version == "theirs":
            assert batch_size is not None
            assert num_batches is not None
            assert num_epochs is None
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.sgcn.parameters()),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=100, gamma=learn_decay
            )
            self.num_batches = num_batches
            self.batch_size = batch_size
            self.adj_lists_pos = self.get_adj_list(train_pos_edge_index)
            self.adj_lists_neg = self.get_adj_list(train_neg_edge_index)
        # the default loss version does full batch gradient descent, and we use Adam with it:
        elif loss_version == "torch-geometric":
            assert num_epochs is not None
            assert num_batches is None
            assert batch_size is None
            self.optimizer = Adam(
                self.sgcn.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
            self.num_epochs = num_epochs
        else:
            raise Exception(f"{loss_version=} is not valid.")
        self.loss_version = loss_version

        # only use training edges for initial SSE
        self.X = self.sgcn.create_spectral_features(
            self.train_pos_edge_index, self.train_neg_edge_index, self.num_nodes
        )

    @staticmethod
    def get_adj_list(edge_index) -> Dict[int, set]:
        adj_list = defaultdict(set)
        for edge in edge_index.T:
            x = edge[0].item()
            y = edge[1].item()
            adj_list[x].add(y)

        return adj_list

    def train(self, plot: bool = False) -> Tensor:

        epochs_since_improvement = 0
        best_embedding: Tensor = torch.zeros((1,))
        best_score = 0

        train_losses = []
        val_aucs = []
        val_f1s = []

        # train model
        if self.loss_version == "theirs":
            done_batches = 0
            epoch = 0
            while done_batches < self.num_batches:  # epochs
                # create batches:
                nodes_perm = torch.randperm(self.num_nodes)  # random nodes for batches
                epoch_train_losses = []
                for start_idx, stop_idx in zip(
                    range(0, self.num_nodes, self.batch_size),
                    range(self.batch_size, self.num_nodes, self.batch_size),
                ):
                    center_nodes = nodes_perm[start_idx:stop_idx].tolist()
                    self.optimizer.zero_grad()
                    loss = self.sgcn.their_loss(
                        center_nodes,
                        self.adj_lists_pos,
                        self.adj_lists_neg,
                        self.num_nodes,
                        self.sgcn(self.X, self.train_pos_edge_index, self.train_neg_edge_index),
                        self.train_pos_edge_index,
                        self.train_neg_edge_index,
                    )
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    epoch_train_losses.append(loss.item())
                    done_batches += 1
                epoch += 1
                train_losses.append(np.average(epoch_train_losses))

                # validation
                if epoch % self.val_interval == 0:
                    embedding, val_auc, val_f1 = self.validate()
                    val_aucs.append(val_auc)
                    val_f1s.append(val_f1)
                    score = val_auc + val_f1
                    if best_score < score:
                        best_embedding = embedding
                        best_score = score
                        epochs_since_improvement = 0
                    else:
                        epochs_since_improvement += 1
                    if epochs_since_improvement > self.early_stopping_patience:
                        break

        else:
            # torch-geometric style, full batch:
            for epoch in range(self.num_epochs):  # epochs
                self.optimizer.zero_grad()
                loss = self.sgcn.loss(
                    self.sgcn(self.X, self.train_pos_edge_index, self.train_neg_edge_index),
                    self.train_pos_edge_index,
                    self.train_neg_edge_index,
                )
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

                if (epoch + 1) % self.val_interval == 0:
                    embedding, val_auc, val_f1 = self.validate()
                    val_aucs.append(val_auc)
                    val_f1s.append(val_f1)
                    score = val_auc + val_f1
                    if best_score < score:
                        best_embedding = embedding
                        best_score = score
                        epochs_since_improvement = 0
                    else:
                        epochs_since_improvement += 1
                    if epochs_since_improvement > self.early_stopping_patience:
                        break

        # plot the learning curves
        if plot:
            x_axis = range(len(train_losses))
            fig, axs = plt.subplots(1, 2)
            axs[0].plot(x_axis, train_losses, label="Training Loss")
            axs[0].set_title("Learning Curve")
            axs[0].legend()
            axs[0].set_xlabel("Epoch")
            axs[0].set_ylabel("Loss")
            axs[0].grid(True)

            x_axis = np.arange(len(val_aucs)) * self.val_interval
            axs[1].plot(x_axis, val_aucs, label="Validation AUC")
            axs[1].plot(x_axis, val_f1s, label="Validation F1")
            axs[1].plot(
                x_axis, [x + y for x, y in zip(val_aucs, val_f1s)], label="Validation F1 + AUC"
            )
            axs[1].grid(True)
            axs[1].set_title("Validation Performance")
            axs[1].legend()
            axs[1].set_xlabel("Epoch")
            axs[1].set_ylabel("Performance")

            plt.show()

        return best_embedding

    def validate(self) -> Tuple[Tensor, float, float]:
        embedding = self.sgcn(self.X, self.train_pos_edge_index, self.train_neg_edge_index)
        auc, f1 = self.sgcn.test(
            embedding,
            self.val_pos_edge_index,
            self.val_neg_edge_index,
        )

        return embedding, auc, f1

    def get_num_parameters(self) -> int:
        return np.sum(p.numel() for p in self.sgcn.parameters() if p.requires_grad)

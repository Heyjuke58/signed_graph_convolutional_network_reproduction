from torch_geometric.nn import SignedGCN
import copy
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable
from torch.optim import Adam
from src.our_signed_gcn import OurSignedGCN


class SGCNTrainer:
    def __init__(
        self,
        # src_dataset,
        train_pos_edge_index: torch.LongTensor,
        train_neg_edge_index: torch.LongTensor,
        test_pos_edge_index: torch.LongTensor,
        test_neg_edge_index: torch.LongTensor,
        num_nodes: int,
        in_features: int,
        out_features: int,
        num_layers: int,
        lamb: float,
        learning_rate: float = 0.01,
        weight_decay: float = 1e-5,
        learn_decay: float = 0.75,
        xent_weights: List[float] = [0.15, 0.8, 0.05],
        activation_fn: Callable = torch.relu,
    ) -> None:
        """
        The edge indices should all be undirected???
        """
        self.train_pos_edge_index = train_pos_edge_index
        self.train_neg_edge_index = train_neg_edge_index
        self.test_pos_edge_index = test_pos_edge_index
        self.test_neg_edge_index = test_neg_edge_index
        self.num_nodes = num_nodes

        self.sgcn = OurSignedGCN(
            in_features,
            out_features,
            num_layers,
            lamb,
            xent_weights=xent_weights,
            activation_fn=activation_fn,
        )
        self.optimizer = Adam(self.sgcn.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # self.optimizer = torch.optim.SGD(
        #     filter(lambda p: p.requires_grad, self.sgcn.parameters()),
        #     lr=learning_rate,
        #     weight_decay=weight_decay,
        # )
        # self.scheduler = torch.optim.lr_scheduler.StepLR(
        #     self.optimizer, step_size=100, gamma=learn_decay
        # )

        # only use training edges for initial SSE
        self.X = self.sgcn.create_spectral_features(
            self.train_pos_edge_index, self.train_neg_edge_index, self.num_nodes
        )

    def train(
        self,
        # optimizer: torch.optim.Optimizer,
        epochs: int,
        early_stopping_patience: int,
        early_stopping: bool = False,
        verbose: bool = False,
        plot: bool = False,
    ) -> Tuple[torch.nn.Module, List[int], List[int]]:

        epochs_since_improvement = 0
        best_val_loss = float("inf")
        best_model = copy.deepcopy(self.sgcn)

        train_losses = []
        test_auc = []
        test_f1 = []

        for epoch in range(epochs):  # epochs
            # train model
            self.optimizer.zero_grad()
            loss = self.sgcn.loss(
                self.sgcn(self.X, self.train_pos_edge_index, self.train_neg_edge_index),
                self.train_pos_edge_index,
                self.train_neg_edge_index,
            )
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()
            train_losses.append(loss.item())

            # test model
            auc, f1 = self.sgcn.test(self.X, self.test_pos_edge_index, self.test_neg_edge_index)
            test_auc.append(auc)
            test_f1.append(f1)

            if verbose:
                print(
                    f"Epoch {epoch + 1:3.0f} Training Loss: {loss.item():.3f}, AUC: {test_auc:.3f}, F1: {test_f1:.3f}"
                )

            # do early stopping to check for overfitting on training data
            # if early_stopping:
            #     if epoch_val_loss < best_val_loss:
            #         best_val_loss = epoch_val_loss
            #         best_model = copy.deepcopy(self.sgcn)
            #         epochs_since_improvement = 0
            #     else:
            #         epochs_since_improvement += 1
            #     if epochs_since_improvement > early_stopping_patience:
            #         self.sgcn = best_model
            #         break

        # plot the learning curves
        if plot:
            x_axis = range(len(train_losses))
            plt.plot(x_axis, train_losses, label="Training Loss")
            plt.plot(x_axis, test_auc, label="Test AUC")
            plt.plot(x_axis, test_f1, label="Test F1")
            plt.title("Learning Curve")
            plt.legend()
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.show()

    # TODO: maybe
    def custom_loss(self, z, M_pos, M_neg):
        raise NotImplementedError


def main():
    pass
    # sgcn_trainer = SGCNTrainer(MY_GRAPH, 3, 3, 2, 5)
    # sgcn_trainer.train(epochs=2, early_stopping_patience=0)


if __name__ == "__main__":
    main()

from src.build_graphs import make_undirected, build_edge_indices
from sklearn.model_selection import train_test_split
from src.train_sgcn import SGCNTrainer
from src.train_sse import SSETrainer
from src.test_embedding import test_embedding
from typing import List, Tuple

BITCOIN_ALPHA_THEIRS = ("bitcoinalpha_theirs.csv", ",")
BITCOIN_OTC_THEIRS = ("bitcoinotc_theirs.csv", ",")
BITCOIN_ALPHA = ("soc-sign-bitcoinalpha.csv", ",")
BITCOIN_OTC = ("soc-sign-bitcoinotc.csv", ",")
SLASHDOT = ("soc-sign-Slashdot090221.txt", "\t")
EPINIONS = ("soc-sign-epinions.txt", "\t")
SLASHDOT_ALTERNATIVE = ("slashdot_zoo.matrix", " ")

ALL_FILES = [BITCOIN_ALPHA, BITCOIN_OTC, SLASHDOT, EPINIONS]
ALL_FILES_REALLY = [
    BITCOIN_ALPHA,
    BITCOIN_ALPHA_THEIRS,
    BITCOIN_OTC,
    BITCOIN_OTC_THEIRS,
    SLASHDOT,
    EPINIONS,
]

# TESTED_DATASET = BITCOIN_OTC
SPLIT_SEED = 1337
TEST_SIZE = 0.2
in_features = 64
out_features = 64
num_layers = 2
lamb = 5
embedding_size = 64
epochs = 200
xent_weights = [0.15, 0.8, 0.05]
learning_rate = 0.5
weight_decay = 0.01
learn_decay = 0.75
# xent_weights = [1, 1, 1]


def main(undirected: bool, datasets: List[Tuple[str, str]]):
    for dataset in datasets:
        print(f"DATASET: {dataset[0]} {xent_weights=}")
        pos_edge_index, neg_edge_index, num_nodes = build_edge_indices(*dataset, directed=True)
        # split dataset, then make the edge indices undirected:
        train_pos_ei, test_pos_ei = train_test_split(
            pos_edge_index.T, test_size=TEST_SIZE, random_state=SPLIT_SEED
        )
        train_neg_ei, test_neg_ei = train_test_split(
            neg_edge_index.T, test_size=TEST_SIZE, random_state=SPLIT_SEED
        )
        if undirected:
            train_pos_ei = make_undirected(train_pos_ei.T)
            train_neg_ei = make_undirected(train_neg_ei.T)
            test_pos_ei = make_undirected(test_pos_ei.T)
            test_neg_ei = make_undirected(test_neg_ei.T)
        else:
            train_pos_ei = train_pos_ei.T
            train_neg_ei = train_neg_ei.T
            test_pos_ei = test_pos_ei.T
            test_neg_ei = test_neg_ei.T

        for lamb in [0, 3, 5, 7, 10]:
            sgcn_trainer = SGCNTrainer(
                train_pos_ei,
                train_neg_ei,
                test_pos_ei,
                test_neg_ei,
                num_nodes,
                in_features,
                out_features,
                num_layers,
                lamb,
                xent_weights=xent_weights,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                learn_decay=learn_decay,
            )

            # train sgcn
            sgcn_trainer.train(epochs=epochs, early_stopping_patience=0, plot=False)
            # get final embedding from sgcn model
            embedding = sgcn_trainer.sgcn(sgcn_trainer.X, train_pos_ei, train_neg_ei)
            # test final embedding on link prediction task
            auc, f1 = test_embedding(
                embedding, embedding_size, train_pos_ei, train_neg_ei, test_pos_ei, test_neg_ei
            )

            print("auc, f1: ", auc, f1)

        # sse_trainer = SSETrainer(
        #     train_pos_ei,
        #     train_neg_ei,
        #     test_pos_ei,
        #     test_neg_ei,
        #     num_nodes,
        #     embedding_size,
        # )

        # embedding = sse_trainer.train()
        # auc, f1 = test_embedding(
        #     embedding, embedding_size, train_pos_ei, train_neg_ei, test_pos_ei, test_neg_ei
        # )

        # print(f"SSE auc: {auc}, f1: {f1} ")


if __name__ == "__main__":
    main(undirected=False, datasets=[BITCOIN_ALPHA, BITCOIN_OTC])

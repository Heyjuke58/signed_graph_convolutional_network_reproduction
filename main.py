from pathlib import Path
from typing import List, Tuple, Type
import time

from sklearn.model_selection import train_test_split
from torch import relu, tanh

from src.build_graphs import build_edge_indices, make_undirected
from src.models import Trainer
from src.models.train_sgcn import SGCNTrainer
from src.models.train_sse import SSETrainer
from src.test_embedding import test_embedding

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
EMBEDDING_SIZE = 64  # how much is used when testing the embedding

ALGORITHMS = [
    (SGCNTrainer, "sgcn2"),
    (SGCNTrainer, "sgcn1"),
    (SGCNTrainer, "sgcn1p"),
    (SSETrainer, "sse"),
]

sgcn2_hyperpars = {
    "in_features": 64,
    "out_features": 64,
    "num_layers": 2,
    "lamb": 5,
    "epochs": 200,
    "xent_weights": [1, 1, 1],  # originally [0.15, 0.8, 0.05]
    "learning_rate": 0.01,  # originally 0.5 (with sgd optimizer and scheduler)
    "weight_decay": 1e-5,  # originally 0.01
    "learn_decay": 0.75,
    "ablation_version": "sgcn2",
    "activation_fn": tanh,
}
sgcn1_hyperpars = sgcn2_hyperpars.copy()
sgcn1_hyperpars.update({"num_layers": 1, "ablation_version": "sgcn1"})
sgcn1p_hyperpars = sgcn2_hyperpars.copy()
sgcn1p_hyperpars.update({"ablation_version": "sgcn1p"})
sse_hyperpars = {
    "embedding_size_true": 128,
    "embedding_size_used": 64,
}

hyperpars = {
    "sgcn2": sgcn2_hyperpars,
    "sgcn1": sgcn1_hyperpars,
    "sgcn1p": sgcn1p_hyperpars,
}


def get_timestamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def main(
    undirected: bool, datasets: List[Tuple[str, str]], algorithms: List[Tuple[Type[Trainer], str]]
):
    results = {}
    for dataset in datasets:
        print(f"DATASET: {dataset[0]}")
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

        for train_class, algorithm in algorithms:

            trainer = train_class(
                train_pos_ei,
                train_neg_ei,
                test_pos_ei,
                test_neg_ei,
                num_nodes,
                **hyperpars[algorithm],
            )

            embedding = trainer.train(verbose=False, plot=False)

            auc, f1 = test_embedding(
                embedding, EMBEDDING_SIZE, train_pos_ei, train_neg_ei, test_pos_ei, test_neg_ei
            )

            results[(algorithm, dataset[0])] = (auc, f1)

    with open(Path("runs") / f"res_{get_timestamp()}.csv", "x") as f:
        f.write("algorithm,dataset,auc,f1\n")
        for key, value in results.items():
            f.write(f"{key[0]},{key[1]},{value[0]},{value[1]}\n")

        # empty row between csv and hyperparamets
        f.write("\n")

        for hyperpars_alg_name, hyperpars_pars in hyperpars.items():
            f.write(f"{hyperpars_alg_name}:\n")
            for hyperpars_par_name, hyperpars_par_value in hyperpars_pars.items():
                f.write(f"\t{hyperpars_par_name}: {hyperpars_par_value}\n")

            # print(f"λ: {lamb}, σ: {activation_fn.__name__}, auc: {auc:.3f}, f1: {f1:.3f} ")

        #     sgcn_trainer = SGCNTrainer(
        #         train_pos_ei,
        #         train_neg_ei,
        #         test_pos_ei,
        #         test_neg_ei,
        #         num_nodes,
        #         in_features,
        #         out_features,
        #         num_layers,
        #         lamb,
        #         xent_weights=xent_weights,
        #         learning_rate=learning_rate,
        #         weight_decay=weight_decay,
        #         learn_decay=learn_decay,
        #         activation_fn=activation_fn,
        #     )

        #     # train sgcn
        #     sgcn_trainer.train(epochs=epochs, early_stopping_patience=0, plot=False)
        #     # get final embedding from sgcn model
        #     embedding = sgcn_trainer.sgcn(sgcn_trainer.X, train_pos_ei, train_neg_ei)
        #     # test final embedding on link prediction task
        #     auc, f1 = test_embedding(
        #         embedding, embedding_size, train_pos_ei, train_neg_ei, test_pos_ei, test_neg_ei
        #     )

        #     print(f"λ: {lamb}, σ: {activation_fn.__name__}, auc: {auc:.3f}, f1: {f1:.3f} ")

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
    main(
        undirected=False,
        datasets=[BITCOIN_ALPHA, BITCOIN_OTC],
        algorithms=[(SGCNTrainer, "sgcn2"), (SGCNTrainer, "sgcn1"), (SGCNTrainer, "sgcn1p")],
    )

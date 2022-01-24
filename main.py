from pathlib import Path
from typing import List, Tuple, Type, Dict, Any
import time
import numpy as np
import torch
from torch import Tensor, LongTensor

from sklearn.model_selection import train_test_split
from torch import relu, tanh

from src.build_graphs import build_edge_indices, make_undirected
from src.models import Trainer
from src.models.train_sgcn import SGCNTrainer
from src.models.train_sse import SSETrainer
from src.test_embedding import test_embedding

DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(DEV)

BITCOIN_ALPHA = {"file": "soc-sign-bitcoinalpha.csv", "split_symbol": ",", "target_num_nodes": None}
BITCOIN_OTC = {"file": "soc-sign-bitcoinotc.csv", "split_symbol": ",", "target_num_nodes": None}
SLASHDOT = {"file": "soc-sign-Slashdot090221.txt", "split_symbol": "\t", "target_num_nodes": 33586}
EPINIONS = {"file": "soc-sign-epinions.txt", "split_symbol": "\t", "target_num_nodes": 16992}

ALL_FILES = [BITCOIN_ALPHA, BITCOIN_OTC, SLASHDOT, EPINIONS]
ALL_FILES_REALLY = [
    BITCOIN_ALPHA,
    # BITCOIN_ALPHA_THEIRS,
    BITCOIN_OTC,
    # BITCOIN_OTC_THEIRS,
    SLASHDOT,
    EPINIONS,
]

# TESTED_DATASET = BITCOIN_OTC
SEED = 1337
TEST_SIZE = 0.2
VAL_SIZE = 0.1
EMBEDDING_SIZE = 64  # how much is used when testing the embedding
UNDIRECTED = True
REPEATS = 3

ALGORITHMS = [
    (SGCNTrainer, "sgcn2"),
    (SGCNTrainer, "sgcn1"),
    (SGCNTrainer, "sgcn1p"),
    (SSETrainer, "sse"),
]

sgcn2_hyperpars_torch = {
    "in_features": 64,
    "out_features": 64,
    "num_layers": 2,
    "lamb": 5,
    "num_epochs": None,
    "num_batches": 200,
    "xent_weights": [1, 1, 1],  # originally [0.15, 0.8, 0.05]
    "learning_rate": 0.01,  # originally 0.5 (with sgd optimizer and scheduler)
    "weight_decay": 1e-5,  # originally 0.01, geometric 1e-5
    "learn_decay": 0.75,
    "ablation_version": "sgcn2",
    "activation_fn": tanh,
    "val_interval": 5,
    "early_stopping_patience": 50,
    "loss_version": "torch-geometric",
}
sgcn2_hyperpars_theirs = {
    "in_features": 64,
    "out_features": 64,
    "num_layers": 2,
    "lamb": 5,
    "num_epochs": None,
    "num_batches": 10000,
    "batch_size": 1000,
    "xent_weights": [0.15, 0.8, 0.05],  # originally [0.15, 0.8, 0.05]
    "learning_rate": 0.5,  # originally 0.5 (with sgd optimizer and scheduler)
    "weight_decay": 0.01,  # originally 0.01, geometric 1e-5
    "learn_decay": 0.75,
    "ablation_version": "sgcn2",
    "activation_fn": tanh,
    "val_interval": 5,
    "early_stopping_patience": 50,
    "loss_version": "theirs",
}
sgcn1_hyperpars = sgcn2_hyperpars_theirs.copy()
sgcn1_hyperpars.update({"num_layers": 1, "ablation_version": "sgcn1"})
sgcn1p_hyperpars = sgcn2_hyperpars_theirs.copy()
sgcn1p_hyperpars.update({"ablation_version": "sgcn1p"})
sse_hyperpars = {
    "embedding_size_true": 128,
    "embedding_size_used": 64,
}

hyperpars = {
    "sgcn2": sgcn2_hyperpars_theirs,
    "sgcn1": sgcn1_hyperpars,
    "sgcn1p": sgcn1p_hyperpars,
    "sse": sse_hyperpars,
}


def get_timestamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def main(
    repeats: int,
    undirected: bool,
    seed: int,
    test_size: float,
    val_size: float,
    embedding_size: int,
    datasets: List[Dict[str, Any]],
    algorithms: List[Tuple[Type[Trainer], str]],
):
    results = {}
    assert test_size + val_size < 1.0
    for dataset in datasets:
        print(f"DATASET: {dataset['file']}")
        pos_edge_index, neg_edge_index, num_nodes = build_edge_indices(
            **dataset, directed=True, seed=seed
        )

        # split test set from training set
        train_pos_ei, test_pos_ei = train_test_split(
            pos_edge_index.T, test_size=test_size, random_state=seed, shuffle=True
        )
        train_neg_ei, test_neg_ei = train_test_split(
            neg_edge_index.T, test_size=test_size, random_state=seed, shuffle=True
        )
        # split validation sets from training set
        train_pos_ei, val_pos_ei = train_test_split(
            train_pos_ei, test_size=val_size / (1 - test_size), random_state=seed, shuffle=True
        )
        train_neg_ei, val_neg_ei = train_test_split(
            train_neg_ei, test_size=val_size / (1 - test_size), random_state=seed, shuffle=True
        )
        if undirected:
            train_pos_ei = make_undirected(train_pos_ei.T).to(DEV)
            train_neg_ei = make_undirected(train_neg_ei.T).to(DEV)
            val_pos_ei = make_undirected(val_pos_ei.T).to(DEV)
            val_neg_ei = make_undirected(val_neg_ei.T).to(DEV)
            test_pos_ei = make_undirected(test_pos_ei.T).to(DEV)
            test_neg_ei = make_undirected(test_neg_ei.T).to(DEV)
        else:
            train_pos_ei = train_pos_ei.T.to(DEV)
            train_neg_ei = train_neg_ei.T.to(DEV)
            val_pos_ei = val_pos_ei.T.to(DEV)
            val_neg_ei = val_neg_ei.T.to(DEV)
            test_pos_ei = test_pos_ei.T.to(DEV)
            test_neg_ei = test_neg_ei.T.to(DEV)

        for train_class, algorithm in algorithms:
            aucs, f1s, runtimes = [], [], []
            for _ in range(repeats):
                start_time = time.perf_counter()

                trainer = train_class(
                    train_pos_ei,
                    train_neg_ei,
                    val_pos_ei,
                    val_neg_ei,
                    test_pos_ei,
                    test_neg_ei,
                    num_nodes,
                    **hyperpars[algorithm],
                )

                print(f"{algorithm=}")
                print(f"{trainer.get_num_parameters()=}")
                embedding = trainer.train(plot=True)

                auc, f1 = test_embedding(
                    embedding, embedding_size, train_pos_ei, train_neg_ei, test_pos_ei, test_neg_ei
                )
                aucs.append(auc)
                f1s.append(f1)
                end_time = time.perf_counter()
                runtimes.append(end_time - start_time)

            results[(algorithm, dataset["file"])] = (
                np.average(aucs),
                np.average(f1s),
                np.average(runtimes),
            )

    with open(Path("runs") / f"res_{get_timestamp()}.csv", "x") as f:
        f.write("algorithm,dataset,auc,f1,avg_runtime\n")
        for key, value in results.items():
            f.write(f"{key[0]},{key[1]},{value[0]},{value[1]},{value[2]}\n")

        # empty row between csv and hyperparamets
        f.write("\n")

        f.write("General hyperparameters:\n")
        f.write(f"\ttest split size: {test_size}\n")
        f.write(f"\tembedding size: {embedding_size}\n")
        f.write(f"\tundirected graph: {undirected}\n")
        f.write(f"\trepeats: {repeats}\n")
        f.write(f"\tseed: {seed}\n")

        f.write("\n")

        for hyperpars_alg_name, hyperpars_pars in hyperpars.items():
            f.write(f"{hyperpars_alg_name}:\n")
            for hyperpars_par_name, hyperpars_par_value in hyperpars_pars.items():
                f.write(f"\t{hyperpars_par_name}: {hyperpars_par_value}\n")


if __name__ == "__main__":
    main(
        repeats=REPEATS,
        undirected=UNDIRECTED,
        seed=SEED,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        embedding_size=EMBEDDING_SIZE,
        # datasets=[BITCOIN_ALPHA, BITCOIN_OTC, EPINIONS, SLASHDOT],
        datasets=[BITCOIN_ALPHA],
        algorithms=[
            (SSETrainer, "sse"),
            (SGCNTrainer, "sgcn2"),
            (SGCNTrainer, "sgcn1"),
            (SGCNTrainer, "sgcn1p"),
        ],
        # algorithms=[
        #     (SGCNTrainer, "sgcn2"),
        # ],
    )

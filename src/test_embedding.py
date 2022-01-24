from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from torch import Tensor, LongTensor
import numpy as np
from typing import Tuple


def test_embedding(
    embedding: Tensor,
    embedding_size: int,
    train_pos_ei: LongTensor,
    train_neg_ei: LongTensor,
    test_pos_ei: LongTensor,
    test_neg_ei: LongTensor,
) -> Tuple[float, float]:
    embedding = embedding[:, :embedding_size]
    model = LogisticRegression(class_weight="balanced", max_iter=400)
    # inner cat combines embs of edges into one feature vector.
    # outer cat combines positive and negative edges:
    embedding = embedding.detach().cpu().numpy()
    train_pos_ei = train_pos_ei.detach().cpu().numpy()
    train_neg_ei = train_neg_ei.detach().cpu().numpy()
    test_pos_ei = test_pos_ei.detach().cpu().numpy()
    test_neg_ei = test_neg_ei.detach().cpu().numpy()
    X_train = np.concatenate(
        (
            np.concatenate(
                (embedding[train_pos_ei[0, :]], embedding[train_pos_ei[1, :]]),
                axis=1,
            ),
            np.concatenate(
                (embedding[train_neg_ei[0, :]], embedding[train_neg_ei[1, :]]),
                axis=1,
            ),
        ),
        axis=0,
    )
    y_train = np.concatenate(
        (np.full(train_pos_ei.shape[1], 1), np.full(train_neg_ei.shape[1], -1))
    )
    X_test = np.concatenate(
        (
            np.concatenate(
                (embedding[test_pos_ei[0, :]], embedding[test_pos_ei[1, :]]),
                axis=1,
            ),
            np.concatenate(
                (embedding[test_neg_ei[0, :]], embedding[test_neg_ei[1, :]]),
                axis=1,
            ),
        ),
        axis=0,
    )
    y_test_true = np.concatenate(
        (np.full(test_pos_ei.shape[1], 1), np.full(test_neg_ei.shape[1], -1))
    )

    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)

    auc = roc_auc_score(y_test_true, y_test_pred)
    f1 = f1_score(y_test_true, y_test_pred)

    return auc, f1

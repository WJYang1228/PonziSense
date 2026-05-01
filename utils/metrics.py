from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def cls_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0),
    }
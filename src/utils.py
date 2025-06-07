import torch

def accuracy_fn(y_true, y_pred):
    # Your accuracy calculation, e.g. for classification:
    correct = (y_true == y_pred).sum().item()
    total = len(y_true)
    return correct / total

# Any other plotting or helper functions

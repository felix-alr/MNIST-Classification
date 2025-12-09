import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def compute_average(model: nn.Module, test_loader: DataLoader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, Y in test_loader:
            pred_labels = torch.argmax(model(X), dim=1)
            ground_truth_labels = torch.argmax(Y, dim=1)
            correct += torch.sum(pred_labels == ground_truth_labels).item()
            total += test_loader.batch_size
        print(f"Correct: {correct}/{total}")
    return correct/total
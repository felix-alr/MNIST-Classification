import types

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from src.test import compute_average


def train_network(model: nn.Module, data_loader: DataLoader, optimizer: optim.Optimizer, loss_fcn: nn.modules.loss._WeightedLoss, options: types.SimpleNamespace):
    optimizer.zero_grad()

    options.epochs = getattr(options, "epochs", 20)
    options.learning_rate = getattr(options, "learning_rate", 1e-2)
    options.validate = getattr(options, "validate", False)
    options.validation_data = getattr(options, "validation_data", None)

    model.train()
    device = next(model.parameters()).device
    for epoch in range(options.epochs):
        print(f"Epoch: {epoch}/{options.epochs}")
        if options.validate and options.validation_data is not None:
            print(f"Accuracy so far: {compute_average(model, options.validation_data)}")
        for batch, (X,Y) in enumerate(data_loader):
            X,Y = X.to(device), Y.to(device)

            pred = model(X)
            loss = loss_fcn(pred, Y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * data_loader.batch_size + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{data_loader.batch_size:>5d}]")
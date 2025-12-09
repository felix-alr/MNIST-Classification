import types

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.model import NeuralNetwork
from src.test import compute_average
from src.train import train_network



from src.gui import DrawingApp
import tkinter as tk

train = False
save_model = True

# Detect device
device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset
training_data = datasets.MNIST(root='./data',
                               train=True,
                               download=True,
                               transform=transforms.ToTensor(),
                               target_transform=transforms.Lambda(
                                   lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor([y]),
                                                                                         1.0)))
test_data = datasets.MNIST(root='./data',
                           train=False,
                           download=True,
                           transform=transforms.ToTensor(),
                           target_transform=transforms.Lambda(
                               lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor([y]), 1.0)))
train_loader = DataLoader(training_data, batch_size=150, shuffle=True, num_workers=4)
test_loader = DataLoader(test_data, batch_size=100, shuffle=False, num_workers=4)

torch.set_num_threads(4)

# Create instance of NeuralNetwork model
model = NeuralNetwork().to(device)


if __name__ == '__main__':
    if train:
        # Define objects for training
        options = types.SimpleNamespace(learning_rate=1e-3)
        options.validate = True
        options.validation_data = test_loader
        options.epochs = 15

        opt_adam = optim.Adam(model.parameters(), lr=options.learning_rate)
        opt_sgdm = optim.SGD(model.parameters(), lr=options.learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        train_network(model, train_loader, opt_adam, loss_fn, options)

        print(compute_average(model, test_loader))
        if save_model:
            torch.save(model.state_dict(), "mnist_model.pth")
    else:
        model.load_state_dict(torch.load("mnist_model.pth", map_location=device))
        root = tk.Tk()
        app = DrawingApp(root, model)
        root.mainloop()
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_curve(x_vals, y_vals, x_label='', y_label='', y_min=None, y_max=None, x_min=None, x_max=None):
    if type(x_vals) == torch.Tensor:
        while len(x_vals.shape) > 1:
            x_vals = x_vals.mean(dim=1)
        x_vals = x_vals.cpu().numpy().tolist()
    if type(y_vals) == torch.Tensor:
        while len(y_vals.shape) > 1:
            y_vals = y_vals.mean(dim=1)
        y_vals = y_vals.cpu().numpy().tolist()
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
    if x_min is not None and x_max is not None:
        plt.xlim(x_min, x_max)
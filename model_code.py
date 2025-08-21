""" Python file containing the code for the Spatio-Temporal Graph Convolutional Networks
architechture, a class for standardizing spatio-temporal tensors, and an MLflow PythonModel
wrapper class for the full model.
"""
import torch
from torch import nn
from torch_geometric_temporal.nn.attention.stgcn import STConv
import numpy as np
import torch.nn.functional as F
import mlflow
from pydantic import BaseModel
from typing import Any, List


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        out = out.permute(0, 2, 3, 1)
        return out


class SpatialTemporalModel(nn.Module):
    def __init__(self, n_nodes, in_channels, n_time_steps_input, H=1,
                 kernel_size=3):
        super(SpatialTemporalModel, self).__init__()
        self.H = H
        self.kernel_size = kernel_size
        
        self.STConv1 = STConv(
            num_nodes=n_nodes,
            in_channels=in_channels,
            hidden_channels=16,
            out_channels=64,
            kernel_size=self.kernel_size,
            K=1
        )

        self.STConv2 = STConv(
            num_nodes=n_nodes,
            in_channels=64,
            hidden_channels=16,
            out_channels=64,
            kernel_size=self.kernel_size,
            K=1
        )

        self.last_temporal = TemporalConv(
            in_channels=64, out_channels=64, kernel_size=self.kernel_size
        )

        linear_size = n_time_steps_input - 2 * (2 * (self.kernel_size - 1)) - self.kernel_size + 1

        self.linear = nn.Linear(linear_size * 64, 1)

        self.softplus = nn.Softplus()

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index, weight_index):
        x = self.STConv1(x, edge_index, weight_index)
        x = self.STConv2(x, edge_index, weight_index)

        x = x.permute(0, 2, 1, 3)
        x = self.last_temporal(x)

        x = self.linear(x.reshape((x.shape[0], x.shape[1], -1)))

        # Ensure outputs are positive
        x = self.softplus(x)
        return x.squeeze(dim=(0, 1))


class TensorStandardizer:
    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, x: torch.Tensor, dim: int | tuple):
        self.means = torch.mean(x, dim=dim)
        self.stds = torch.std(x, dim=dim)

    def transform(self, x: torch.Tensor):
        return (x - self.means) / self.stds

    def fit_transform(self, x: torch.Tensor, dim: int | tuple):
        self.fit(x, dim)
        return self.transform(x)

    def save_means_stds(self, means_path, stds_path):
        if self.means is not None:
            np.save(means_path, self.means.numpy())

        if self.stds is not None:
            np.save(stds_path, self.means.numpy())

    def load_means_stds(self, means_path, stds_path):
        self.means = torch.tensor(np.load(means_path), dtype=torch.float64)
        self.stds = torch.tensor(np.load(stds_path), dtype=torch.float64)


class GraphInput(BaseModel):
    """Input for GraphModel"""
    x: list[list[list[float]]]
    # edge_index: list[list[int]]
    # edge_attr: list[float]


class GraphModel(mlflow.pyfunc.PythonModel):
    """
    Wraps SpatialTemporalModel into a pyfunc PythonModel
    """
    def __init__(self, model, standardizer):
        self.model = model
        self.model.eval()
        self.standardizer = standardizer
        self.edge_index = None
        self.edge_attr = None

    def predict(self, context, model_input: List[GraphInput], params=None) -> list[list[Any]]:
        """
        :param context:
        :param model_input: list[dict[str, float]] formatted as {
            'x': nested lists shaped (n_time_steps, n_nodes, n_features),
        }
        :param params:
        :return: np.ndarray
        """
        predictions = []
        for item in model_input:
            x = item.x
            x = torch.tensor(x, dtype=torch.float).unsqueeze(0)
            x = self.standardize(x)

            edge_index = torch.tensor(self.edge_index, dtype=torch.int64)
            edge_attr = torch.tensor(self.edge_attr)

            with torch.no_grad():
                out = self.model(x, edge_index, edge_attr)
                predictions.append(out.numpy().tolist())
        return predictions

    def load_context(self, context):
        self.edge_index = np.load(context.artifacts["edges_npy"]).T
        self.edge_attr = np.load(context.artifacts["edge_weights_npy"])

    def standardize(self, x: torch.Tensor) -> torch.Tensor:
        x[:, :, :, :11] = self.standardizer.transform(x[:, :, :, :11])
        return x
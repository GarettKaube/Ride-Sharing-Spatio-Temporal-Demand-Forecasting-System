""" Training script for Spatial Temporal Conv network. Uses MLflow to log 
metrics and wraps the model in a mlflow.pyfunc.PythonModel 
for easy logging and serving
args: 
    --bacth_size: int
    --n_epochs: int, default 50
    --mode: ['train', 'test'], default: 'train'
"""
from model_code.model_code import SpatialTemporalModel, TensorStandardizer, GraphModel
from src.forecast import lag_features_np
import pandas as pd
import numpy as np

import torch
from torch import nn
from torch_geometric_temporal.signal import StaticGraphTemporalSignal, temporal_signal_split

from contextlib import nullcontext

from tqdm import tqdm
from torch.nn import PoissonNLLLoss
import time

import mlflow
from mlflow.models.signature import infer_signature
import os
import sys
import random
import argparse
from torch.utils.data import Dataset

from typing import Literal
from pathlib import Path

TRACKING_URI = os.environ['MLFLOW_TRACKING_URI']
N_TIME_STEPS_BACK = 15

path = Path(__file__).resolve().parent

sys.path.insert(0, str(path))


def parse_args():
    argparsser = argparse.ArgumentParser()
    argparsser.add_argument("--batch_size", dest="batch_size", type=int)
    argparsser.add_argument("--n_epochs", dest="n_epochs", type=int, default=50)
    argparsser.add_argument("--mode", dest="mode", type=str, default="train", choices=['train', 'test'])
    args = argparsser.parse_args()
    return args


class TemporalGraphDataset(Dataset):
    def __init__(self, dataset: StaticGraphTemporalSignal):
        self.features = dataset.features.copy()
        self.targets = dataset.targets.copy()
        self.edge_index = torch.tensor(dataset.edge_index.copy(), dtype=torch.long)
        self.edge_attr = torch.tensor(dataset.edge_weight.copy())

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'x': torch.tensor(self.features[idx], dtype=torch.float),
            'y': torch.tensor(self.targets[idx], dtype=torch.float),
            'edge_index': self.edge_index,
            'edge_attr': self.edge_attr
        }


class Trainer:
    def __init__(
        self,
        n_features,
        model:torch.nn.Module,
        optimizer,
        train_loader,
        validation_loader,
        n_time_steps_back,
        saved_model_path,
        checkpoints_path,
        check_point_interval=2,
    ):
        self.model = model
        self.n_features = n_features
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.n_time_steps_back = n_time_steps_back

        self.mse = nn.MSELoss()
        self.loss = PoissonNLLLoss(log_input=False)

        self.mse_train = []
        self.mse_val = []
        self.poisson_loss_train = []
        self.poisson_loss_val = []

        self.saved_model_path = saved_model_path
        self.checkpoints_path = checkpoints_path

        self.best_val_loss = 1000000000
        self.best_val_epoch = 0

        self.check_point_interval = check_point_interval

    def train(self, n_epochs:int) -> None:
        """ Runs the training process and validation for each epoch
        """
        for epoch in tqdm(range(n_epochs)):
            self.model.train()
            train_loss = self._step(epoch)
            self.mse_train.append(train_loss[1])
            self.poisson_loss_train.append(train_loss[0])

            print("Validating")
            val_loss = self._step(epoch, mode="validation")
            self.mse_val.append(val_loss[1])
            self.poisson_loss_val.append(val_loss[0])

    def _step(self, epoch:int, mode: Literal['train', 'validation'] = 'train') -> None:
        cost = 0
        mse_total = 0

        dataloader = self.validation_loader if mode == "validation" else self.train_loader

        if mode == "validation":
            self.model.eval()

        with (torch.no_grad() if mode=="validation" else nullcontext()):
            for timestamp, snapshot in enumerate(dataloader):
                if mode == "train":
                    self.optimizer.zero_grad()

                x = snapshot['x']

                n_nodes = x.shape[2]
                n_features = x.shape[-1]

                if x.dim() == 2:
                    x = x.permute(1, 0).unsqueeze(-1)

                x = x.view(-1, self.n_time_steps_back + 1, n_nodes, n_features)

                edge_index = snapshot['edge_index'][0, :, :]
                edge_attr = snapshot['edge_attr'][0, :]

                start = time.time()
                y_hat = self.model(x, edge_index, edge_attr)

                # Get a time for how fast each forward pass is
                if epoch == 0 and timestamp == 0:
                    end = time.time()
                    print(f"Elapsed time: {end - start:.4f} seconds")

                y = snapshot['y']
                y = y.view(-1, 77, 1).squeeze()
                batch_loss = self.loss(y_hat.squeeze(), y)
                cost = cost + batch_loss.item()

                mse_batch = self.mse(y_hat.squeeze(), y)
                mse_total = mse_total + mse_batch.item()

                if mode == "train":
                    mse_batch.backward()
                    self.optimizer.step()

            if mode == "validation" and mse_total < self.best_val_loss:
                print("Saving current best model")
                torch.save(self.model.state_dict(), self.saved_model_path)
                self.best_val_loss = mse_total
                self.best_val_epoch = epoch

            if mode == "validation" and ((epoch + 1) % self.check_point_interval == 0):
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': mse_total
                }, f'{self.checkpoints_path}/checkpoint_{epoch}.pth'
                )

        cost = cost / (timestamp + 1)
        print(f"NNL {mode}: {cost}")
        print(f"MSE {mode}: {mse_total / (timestamp + 1)}")

        return cost, mse_total


def fill_absent_communities(X):
    weather_features = ['temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'pres', 'coco']
    community_areas = set(range(1, 77 + 1))

    missing_com = community_areas - set(X['pickup_community_area'])

    day_of_week_sin = X["trip_start_day_of_week_sin"].iat[0]
    day_of_week_cos = X["trip_start_day_of_week_cos"].iat[0]

    hour_sin = X["trip_start_hour_sin"].iat[0]
    hour_cos = X["trip_start_hour_cos"].iat[0]

    def fill_val(val=None):
        if val:
            return [val for com in missing_com]

        return [0 for com in missing_com]

    columns = X.columns.drop(
        ["pickup_community_area",
         "trip_start_day_of_week_sin",
         "trip_start_day_of_week_cos",
         "trip_start_hour_sin",
         "trip_start_hour_cos"] + weather_features
    )

    data = {
        "pickup_community_area": [com for com in missing_com],
        "trip_start_day_of_week_sin": fill_val(day_of_week_sin),
        "trip_start_day_of_week_cos": fill_val(day_of_week_cos),
        "trip_start_hour_sin": fill_val(hour_sin),
        "trip_start_hour_cos": fill_val(hour_cos)
    }

    # Fill in the weather
    for weather_fet in weather_features:
        data.update({weather_fet: fill_val(X[weather_fet].iloc[-1])})

    # Fill in other unknown columns with 0
    more_data = {
        col: fill_val() for col in columns
    }

    data.update(more_data)

    new_df = pd.DataFrame(data)

    # Make sure pickup communities always in correct order
    X = pd.concat([X, new_df], axis=0, ignore_index=True) \
        .sort_values(by="pickup_community_area")
    return X


def make_features_array(df:pd.DataFrame) -> tuple[list[np.ndarray], list[np.ndarray]]:
    groups = ['interval']
    graph_ts = df.groupby(groups)[df.columns.drop(groups)]
    features = []
    targets = []
    for g, val in graph_ts:
        val = fill_absent_communities(val)

        feat = val.drop(['pickup_community_area', 'target'], axis=1) \
            .to_numpy()

        trgts = val['target'].to_numpy()

        features.append(feat)
        targets.append(trgts)

    return features, targets


def main():
    args = parse_args()

    batch_size = args.batch_size
    n_epochs = args.n_epochs

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("rides_forecasting")

    df = pd.read_csv("data.csv", index_col=0)

    features, targets = make_features_array(df)
    features = np.array(features, dtype=float)

    added_lagged_features = lag_features_np(features, N_TIME_STEPS_BACK)

    edges_np = np.load("./graph_files/edges.npy")
    edge_weights = np.load("./graph_files/edge_weights.npy")


    dataset = StaticGraphTemporalSignal(
        edge_index=edges_np.T,
        edge_weight=edge_weights,
        features=added_lagged_features,
        targets=targets[N_TIME_STEPS_BACK:],
    )


    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.80)
    train_dataset, validation = temporal_signal_split(train_dataset, train_ratio=0.80)

    standardizer = TensorStandardizer()
    features_tensor = torch.tensor(train_dataset.features.copy())
    # up to 11 continuous variables
    train_dataset.features[:, :, :, :11] = standardizer.fit_transform(features_tensor[:, :, :, :11], dim=(0, 1))
    validation.features[:, :, :, :11] = standardizer.transform(torch.tensor(validation.features.copy()[:, :, :, :11]))
    test_dataset.features[:, :, :, :11] = standardizer.transform(
        torch.tensor(test_dataset.features.copy()[:, :, :, :11]))


    train_torch_dataset = TemporalGraphDataset(train_dataset)
    validation_torch_dataset = TemporalGraphDataset(validation)
    test_torch_dataset = TemporalGraphDataset(test_dataset)

    train_loader = torch.utils.data.DataLoader(train_torch_dataset, batch_size=batch_size, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(validation_torch_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_torch_dataset, batch_size=batch_size, shuffle=False)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    name = "SpatialTemporalGNN" + str(random.randint(10000, 999999))

    with mlflow.start_run(run_name=name) as run:
        n_features = train_dataset.features.shape[-1]
        n_time_steps_input = 16

        model = SpatialTemporalModel(
            in_channels=n_features, n_nodes=77, n_time_steps_input=n_time_steps_input
        )

        lr = 0.001
        weight_decay = 1e-4

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        mlflow.log_params({
            "lr": lr,
            "weight_decay": weight_decay,
            "n_time_steps_input": n_time_steps_input,
            "n_features": n_features,
            "batch_size": batch_size,
            "n_epochs": n_epochs
        })

        trainer = Trainer(
            n_features=n_features,
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            validation_loader=validation_loader,
            check_point_interval=2,
            saved_model_path="models/SpatialTemporalv1.pth",
            checkpoints_path="./model_checkpoints",
            n_time_steps_back=N_TIME_STEPS_BACK
        )

        start = time.time()
        #trainer.train(n_epochs=n_epochs)
        end = time.time()

        train_duration = end - start
        print(f"train duration: {train_duration}")

        mlflow.log_metrics({
            "BestValidationMSE": trainer.best_val_loss,
            "BestValidationEpoch": trainer.best_val_epoch,
        })

        mlflow.log_dict({
            "PoissonNLL": trainer.poisson_loss_train,
            "ValidationPoissonNLL": trainer.poisson_loss_val,
            "MSE": trainer.mse_train,
            "ValidationMSE": trainer.mse_val
        }, "performance.json")

        for i in train_loader:
            print(i['x'][0, :, :, :].squeeze(0).shape)
            input_example = [{
                'x': i['x'][0, :, :, :].squeeze(0).numpy().tolist(),
                'edge_index': i['edge_index'][0, :, :].squeeze(0).numpy().tolist(),
                'edge_attr': i['edge_attr'][0, :].squeeze(0).numpy().tolist()
            }]
            break

        # Log best model
        best_model = SpatialTemporalModel(in_channels=n_features, n_time_steps_input=16, n_nodes=77)
        best_model.load_state_dict(torch.load("models/SpatialTemporalv1.pth", weights_only=True))

        best_model_pyfunc = GraphModel(model, standardizer)

        out = best_model_pyfunc.predict(context=None, model_input=input_example)

        sig = infer_signature(model_input=input_example,
                              model_output=np.array(out[0]))


        mlflow.pyfunc.log_model(
            python_model=best_model_pyfunc,
            artifact_path="model",
            pip_requirements=[
                "torch==2.6.0",
                "torchvision==0.21.0",
                "torch-geometric==2.6.1",
                "torch_geometric_temporal==0.56.1",
                "torch_sparse==0.6.18",
                "torch_scatter==2.1.2",
                "numpy==1.26.4"
            ],
            signature=sig,
            code_paths=['model_code']
        )

if __name__ == "__main__":
    main()
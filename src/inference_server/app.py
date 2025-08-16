from pydantic import BaseModel
import logging
from fastapi import FastAPI
import torch
import numpy as np
from model_code.model_code import SpatialTemporalModel, TensorStandardizer
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(debug=True)


class Request(BaseModel):
    date: list[str]
    X: list
    communities: list[int] | int | None = None



async def get_model():
    weights = torch.load("models/SpatialTemporalv1.pth", weights_only=True)
    model = SpatialTemporalModel(in_channels=15, n_time_steps_input=16, n_nodes=77)
    model.load_state_dict(weights)
    model.eval()

    # Load graph adjacency matrix and edge weights
    edges = np.load("./graph/edges.npy")
    edge_weights = np.load("./graph/edge_weights.npy")

    standardizer = TensorStandardizer()
    means_path = "./standardizer/standardizer_means.npy"
    std_path = "./standardizer/standardizer_stds.npy"
    standardizer.load_means_stds(means_path, std_path)

    return model, standardizer, edges, edge_weights


def standardize_features(X, standardizer):
    X.features[:, :, :, :11] = standardizer.transform(torch.tensor(X.features.copy()[:, :, :, :11]))
    return X


def validate_dims(X):
    if X.ndim != 4:
        raise ValueError(f"Did not get dimension 4 input. Got dim: {X.ndim}")


def generate_forecasts(data, model, communities):
    predictions = []
    with torch.no_grad():
        for idx, item in enumerate(data):
            x = item.x.unsqueeze(0)
            out = model(x, item.edge_index, item.edge_attr)
            print(out.shape)
            if communities is not None:
                predictions.append(out.numpy()[communities].tolist())
    return predictions


def inference(
        request,
        model,
        standardizer,
        edges,
        edge_weights,
):
    # Store model predictions
    result = {}

    X = np.array(request.X)
    validate_dims(X)

    # If we want to keep a subset of communities for the forecasts
    communities = request.communities

    data = StaticGraphTemporalSignal(
        edge_index=edges.T,
        edge_weight=edge_weights,
        features=X,
        targets=[None] * X.shape[0]
    )

    data = standardize_features(data, standardizer)

    # Use the model to generate predictions
    predictions = generate_forecasts(data, model, communities)

    result["request"] = request.model_dump()
    result['forecast'] = predictions

    return result


@app.post("/invocations/", status_code=200)
async def return_forecasts(request:Request):
    model, standardizer, edges, edge_weights = await get_model()

    result = inference(request, model, standardizer, edges, edge_weights)
    return result



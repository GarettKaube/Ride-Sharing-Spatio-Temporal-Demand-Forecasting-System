import pytest
import torch
from model_code.model_code import SpatialTemporalModel

# Params are (num_lags + 1, num_nodes, num_features)
@pytest.fixture(params=[(15, 77, 10)])
def sample_input(input_shape):
    input_shape = input_shape.param
    tensor = torch.randn(size = (100, input_shape[0], input_shape[1], input_shape[2]))
    return tensor


test_data = [
    (10, 5, 20, 1, 2)
]

@pytest.mark.parametrize("n_nodes,in_channels,n_time_steps_input,H,kernel_size", test_data)
def test_SpatialTemporalModel(n_nodes,in_channels,n_time_steps_input,H,kernel_size):

    model = SpatialTemporalModel(n_nodes,in_channels,n_time_steps_input,H,kernel_size)

    assert model.H == H
    assert model.kernel_size == kernel_size

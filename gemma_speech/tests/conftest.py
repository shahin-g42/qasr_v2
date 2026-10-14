import pytest
import torch


@pytest.fixture(autouse=True)
def limit_cpu_threads():
    torch.set_num_threads(1)
    torch.manual_seed(123)

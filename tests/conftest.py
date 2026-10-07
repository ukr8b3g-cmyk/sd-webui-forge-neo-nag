import pytest
import torch


@pytest.fixture(autouse=True)
def cpu_seed():
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(72421)
        with torch.inference_mode():
            yield

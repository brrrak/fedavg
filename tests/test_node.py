import torch
import torchvision

from fedavg.node import Node
from fedavg.model import Net
import copy


def test_aggregation():
    sw = Node(0, model=Net())
    cli1 = Node(1, model=Net())
    cli2 = Node(2, model=Net())
    cli3 = Node(3, model=Net())

    cli1.model.state_dict()["fully_connected.2.bias"] *= 0  # make it all zeros
    cli2.model.state_dict()["fully_connected.2.bias"] *= 0  # make it all zeros
    cli2.model.state_dict()["fully_connected.2.bias"] += 1  # make it all ones
    cli3.model.state_dict()["fully_connected.2.bias"] *= 0  # make it all zeros
    cli3.model.state_dict()["fully_connected.2.bias"] += 2  # make it all twos

    sw.add_client(cli1)
    sw.add_client(cli2)
    sw.add_client(cli3)

    initial_model = copy.deepcopy(sw.model.state_dict())
    prior_last_layer = initial_model["fully_connected.2.bias"]
    sw.aggregate([cli.model for cli in sw.clients])
    last_layer_agg = sw.model.state_dict()["fully_connected.2.bias"]

    is_changed = (last_layer_agg - prior_last_layer).abs() < 1e-9
    assert any(is_changed) is False

    is_eq = (last_layer_agg - 1).abs() < 1e-9
    assert all(is_eq) is True


def test_training():
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=64, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=64, shuffle=True)

    node = Node(
        node_id=0,
        model=Net(),
        clients=None,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=torch.nn.CrossEntropyLoss(),
    )

    node.train(local_updates=1)

    assert node.test() > 85

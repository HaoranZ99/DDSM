from torch_geometric.datasets import (
    Planetoid,
    CoraFull,
    Coauthor,
    WebKB,
    WikipediaNetwork,
    WikiCS,
)
from ddsm import DDSM


def get_model(model: str, dataset, args):
    if model == "DDSM":
        model = DDSM(
            in_channels=dataset.data.num_features,
            hidden_channels=args["hidden_dim"],
            out_channels=dataset.num_classes,
            num_layers=args["num_layers"],
            alpha=args["alpha"],
            beta=args["beta"],
            eta=args["eta"],
            gamma=args["gamma"],
            dropout=args["dropout"],
        )
    return model


def get_dataset(root: str, name: str):
    if name in ["Cora", "CiteSeer", "PubMed"]:
        dataset = Planetoid(root=root, name=name)
    elif name == "CoraFull":
        dataset = CoraFull(root=root)
    elif name in ["CS", "Physics"]:
        dataset = Coauthor(root=root, name=name)
    elif name in ["Cornell", "Texas", "Wisconsin"]:
        dataset = WebKB(root=root, name=name)
    elif name == "Chameleon":
        dataset = WikipediaNetwork(root=root, name=name.lower())
    elif name == "WikiCS":
        dataset = WikiCS(root=root, is_undirected=True)
    else:
        raise Exception("Unknown dataset.")
    return dataset

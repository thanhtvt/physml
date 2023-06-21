import argparse
import os
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import VirtualNode

from src.nn.configs import graph as conf
from src.nn.datasets import GraphDataset, QM7Dataset
from src.nn.models import MPNN
from src.nn.trainers import GraphTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--devices",
        type=str,
        default="-1",
        help="Separated by comma, e.g. 0,1,2,3",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help="Fold to use for training",
    )
    parser.add_argument(
        "--new-wandb",
        action="store_true",
        help="Create a new wandb run",
    )
    args = parser.parse_args()
    return args


def init_train(device, fold: int):
    # Load dataloader
    train_dataset = GraphDataset(root=conf.SAVE_DATA_DIR,
                                 transform=VirtualNode(),
                                 data_dir=conf.DATA_DIR,
                                 fold=fold,
                                 train=True)
    test_dataset = GraphDataset(root=conf.SAVE_DATA_DIR,
                                transform=VirtualNode(),
                                data_dir=conf.DATA_DIR,
                                fold=fold,
                                train=False)

    # train_dataset = QM7Dataset(root=conf.SAVE_DATA_DIR,
    #                            transform=VirtualNode(),
    #                            data_dir=conf.DATA_DIR,
    #                            fold=fold,
    #                            train=True)
    # test_dataset = QM7Dataset(root=conf.SAVE_DATA_DIR,
    #                           transform=VirtualNode(),
    #                           data_dir=conf.DATA_DIR,
    #                           fold=fold,
    #                           train=False)
    # train_dataset = QM7Dataset(root=conf.SAVE_DATA_DIR, transform=VirtualNode())
    # test_dataset = QM7Dataset(root=conf.SAVE_DATA_DIR, transform=VirtualNode())

    train_loader = DataLoader(train_dataset,
                              batch_size=conf.BATCH_SIZE,
                              num_workers=conf.NUM_WORKERS,
                              drop_last=False,
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=conf.BATCH_SIZE,
                             num_workers=conf.NUM_WORKERS,
                             drop_last=False,
                             shuffle=False)

    # Initialize model
    model = MPNN(
        node_input_dim=conf.NODE_INPUT_DIM,
        edge_input_dim=conf.EDGE_INPUT_DIM,
        gnn_node_output_dim=conf.GNN_NODE_OUTPUT_DIM,
        gnn_edge_hidden_dim=conf.GNN_EDGE_HIDDEN_DIM,
        num_step_message_passing=conf.NUM_STEP_MESSAGE_PASSING,
        readout_node_hidden_dim=conf.READOUT_NODE_HIDDEN_DIM,
        readout_node_output_dim=conf.READOUT_NODE_OUTPUT_DIM,
        pooling_mode=conf.POOLING_MODE,
        output_dim=conf.OUTPUT_DIM,
    ).to(device)

    return model, train_loader, test_loader


def get_optimizer(params):
    if conf.OPTIMIZER == "adamw":
        return torch.optim.AdamW(
            params=params,
            lr=conf.LEARNING_RATE,
            weight_decay=conf.WEIGHT_DECAY,
        )
    elif conf.OPTIMIZER == "adam":
        return torch.optim.Adam(
            params=params,
            lr=conf.LEARNING_RATE,
            weight_decay=conf.WEIGHT_DECAY,
        )
    elif conf.OPTIMIZER == "sgd":
        return torch.optim.SGD(
            params=params,
            lr=conf.LEARNING_RATE,
            weight_decay=conf.WEIGHT_DECAY,
            momentum=conf.MOMENTUM,
        )
    else:
        raise ValueError(f"Unknown optimizer: {conf.OPTIMIZER}")


def train(args):
    # Initialize model & dataloaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_dataloader, test_dataloader = init_train(device, args.fold)

    # Initialize scheduler
    optimizer = get_optimizer(model.parameters())
    scheduler = None

    # Resume WandB run
    resume = "never" if args.new_wandb else "auto"

    # Initialize trainer
    trainer = GraphTrainer(
        config=conf,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        resume=resume,
        checkpoint_name=f"graph_{args.fold}.pt",
    )

    trainer.fit(train_dataloader, test_dataloader)


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    train(args)

import argparse
import os
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import VirtualNode

from src.nn.configs import graph as conf
from src.nn.datasets import GraphDataset
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
    parser.add_argument(
        "--resume-training",
        action="store_true",
        help="Continue training from a checkpoint",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate model on test set",
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
        gnn_dropout_rate=conf.GNN_DROPOUT_RATE,
        num_step_message_passing=conf.NUM_STEP_MESSAGE_PASSING,
        readout_node_hidden_dim=conf.READOUT_NODE_HIDDEN_DIM,
        readout_node_output_dim=conf.READOUT_NODE_OUTPUT_DIM,
        readout_dropout_rate=conf.READOUT_DROPOUT_RATE,
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
    checkpoint_name = args.checkpoint_name or f"graph_{conf.LOGNAME}_{args.fold}.pt"
    trainer = GraphTrainer(
        config=conf,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        resume=resume,
        checkpoint_name=checkpoint_name,
        test=args.test,
    )
    if args.resume_training:
        trainer.load_model()

    trainer.fit(train_dataloader, test_dataloader)


def test(args):
    # Initialize model & dataloaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, test_dataloader = init_train(device, args.fold)

    # Initialize scheduler
    optimizer = get_optimizer(model.parameters())
    scheduler = None

    # Resume WandB run
    resume = "never" if args.new_wandb else "auto"

    # Initialize trainer
    checkpoint_name = args.checkpoint_name or f"graph_{conf.LOGNAME}_{args.fold}.pt"
    trainer = GraphTrainer(
        config=conf,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        resume=resume,
        checkpoint_name=checkpoint_name,
        test=args.test,
    )
    trainer.load_model()

    test_loss, test_mae = trainer.evaluate(test_dataloader)
    print(f"Test loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    if args.test:
        test(args)
    else:
        train(args)

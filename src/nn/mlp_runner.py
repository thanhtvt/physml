import argparse
import os
import torch
from torch.utils.data import DataLoader

from src.nn.configs import mlp as conf
from src.nn.datasets import MLPDataset
from src.nn.models import MultiLayerPerceptron
from src.nn.trainers import MLPTrainer
from src.utils.qm7 import load_qm7
from src.utils.common import split_dictionary


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
        "--feature-type",
        type=str,
        default="coulomb",
        choices=["coulomb", "sorted", "eigen"],
        help="Type of feature to use",
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
        "--test",
        action="store_true",
        help="Evaluate model on test set",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    args = parser.parse_args()
    return args


def get_dataloader(data, feature_type: str):
    if feature_type == "eigen":
        inp_stats = (conf.EIGEN_MEAN, conf.EIGEN_STD)
    else:
        inp_stats = (conf.COULOMB_MEAN, conf.COULOMB_STD)

    dataset = MLPDataset(
        data,
        data_mean=(inp_stats[0], conf.ENERGY_MEAN),
        data_std=(inp_stats[1], conf.ENERGY_STD),
        feature_type=feature_type,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=conf.BATCH_SIZE,
        shuffle=True,
        num_workers=conf.NUM_WORKERS,
    )
    return dataloader


def init_train(feature_type, device, fold: int, test: bool = False):
    # Load dataloader
    data_train, data_test = load_qm7(conf.DATA_DIR, fold=fold)
    data_train, data_val = split_dictionary(data_train, ratio=conf.TRAIN_RATIO)
    train_dataloader = get_dataloader(data_train, feature_type)
    if test:
        valid_dataloader = get_dataloader(data_test, feature_type)
    else:
        valid_dataloader = get_dataloader(data_val, feature_type)

    # Initialize model
    input = train_dataloader.dataset.__getitem__(0)[0]
    input_size = input.shape[-1]
    model = MultiLayerPerceptron(
        input_size=input_size,
        output_size=1,
        hidden_sizes=conf.HIDDEN_SIZES,
        dropout_rate=conf.DROPOUT_RATE,
        activation_type=conf.ACTIVATION,
    )
    model = model.to(device)
    return model, train_dataloader, valid_dataloader


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
    model, train_dataloader, val_dataloader = init_train(
        feature_type=args.feature_type, device=device, fold=args.fold, test=args.test
    )

    # Initialize scheduler
    optimizer = get_optimizer(model.parameters())
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=conf.EPOCHS,
    )

    # Resume WandB run
    resume = "never" if args.new_wandb else "auto"

    # Initialize trainer
    trainer = MLPTrainer(
        config=conf,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        resume=resume,
        checkpoint_name=f"mlp_{args.feature_type}_{args.fold}.pt",
    )
    if args.resume_training:
        trainer.load_model()

    trainer.fit(train_dataloader, val_dataloader)


def test(args):
    # Initialize model & dataloaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, test_dataloader = init_train(
        feature_type=args.feature_type, device=device, fold=args.fold
    )

    # Initialize scheduler
    optimizer = get_optimizer(model.parameters())
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=conf.EPOCHS,
    )

    # Resume WandB run
    resume = "never" if args.new_wandb else "auto"

    # Initialize trainer
    checkpoint_name = args.checkpoint_name or f"mlp_{args.feature_type}_{args.fold}.pt"
    trainer = MLPTrainer(
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

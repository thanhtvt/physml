import argparse
import logging
import os
import torch
# import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import MLPDataset
from models import MultiLayerPerceptron
from configs import dnn as conf
from src.utils.qm7 import load_qm7
from src.utils.common import split_dictionary

logging.basicConfig(filename=conf.LOGFILE,
                    filemode="a",
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%d-%b-%y %H:%M:%S",
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_dataloader(config, data, max_value=None):
    dataset = MLPDataset(
        data=data,
        noise=config.NOISE,
        step=config.EXPAND_STEPS,
        max_value=max_value,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    return dataloader


def get_optimizer(config, params):
    if config.OPTIMIZER == "adamw":
        return torch.optim.AdamW(
            params=params,
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
    elif config.OPTIMIZER == "adam":
        return torch.optim.Adam(
            params=params,
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
    elif config.OPTIMIZER == "sgd":
        return torch.optim.SGD(
            params=params,
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            momentum=config.MOMENTUM,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.OPTIMIZER}")


def train_one_epoch(config, model, dataloader, loss_fn, optimizer, device):
    running_loss = 0.0
    last_loss = 0.0
    model.train()
    for batch_idx, (X, y) in enumerate(dataloader):
        X = X.squeeze(1).to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        running_loss = running_loss + loss.item()
        if batch_idx % config.LOG_INTERVAL == (config.LOG_INTERVAL - 1):
            last_loss = running_loss / config.LOG_INTERVAL
            tqdm.write(f"\tBatch {batch_idx + 1} Loss: {last_loss:.4f}")
            logger.debug(f"\tBatch {batch_idx + 1} Loss: {last_loss:.4f}")
            running_loss = 0.0

    return last_loss


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.squeeze(1).to(device)
            y = y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            running_loss = running_loss + loss.item()

    return running_loss / len(dataloader)


def train(config, device, num_folds: int = 5):
    # init
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.LOGFILE), exist_ok=True)
    os.system(f"touch {config.LOGFILE}")
    # wandb.init(project="chemml", config=config)

    for fold in range(num_folds):
        train_single_fold(config, device, fold)


def train_single_fold(config, device, fold: int):
    # load dataloader
    data_train, data_test = load_qm7(config.DATA_DIR, fold=fold)
    data_train, data_val = split_dictionary(data_train, ratio=config.TRAIN_RATIO)
    train_dataloader = get_dataloader(config, data_train)
    train_max_value = train_dataloader.dataset.get_max()
    valid_dataloader = get_dataloader(config, data_val, train_max_value)
    test_dataloader = get_dataloader(config, data_test, train_max_value)

    input = train_dataloader.dataset.__getitem__(0)[0]
    input_size = input.shape[-1]

    # load model
    model = MultiLayerPerceptron(
        input_size=input_size,
        output_size=1,
        hidden_sizes=config.HIDDEN_SIZES,
    )
    model = model.to(device)

    # load optimizer
    optimizer = get_optimizer(config, model.parameters())

    # load loss function
    loss_fn = torch.nn.L1Loss()
    best_loss = 1e6

    # train
    for epoch in tqdm(range(config.EPOCHS), desc="Epoch"):
        tqdm.write(f"Epoch {epoch + 1}")
        logger.debug(f"Epoch {epoch + 1}")
        avg_loss = train_one_epoch(config, model, train_dataloader, loss_fn, optimizer, device)
        avg_val_loss = evaluate(model, valid_dataloader, loss_fn, device)
        tqdm.write(f"TRAIN: {avg_loss:.4f}\tVALID: {avg_val_loss:.4f}")
        logger.debug(f"TRAIN: {avg_loss:.4f}\tVALID: {avg_val_loss:.4f}")
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model, os.path.join(config.CHECKPOINT_DIR, "mlp.pt"))

    # test
    model = torch.load(os.path.join(config.CHECKPOINT_DIR, "mlp.pt"))
    avg_test_loss = evaluate(model, test_dataloader, loss_fn, device)
    tqdm.write(f"TEST: {avg_test_loss:.4f}")
    logger.debug(f"TEST: {avg_test_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--devices",
        type=str,
        default="-1",
        help="Separated by comma, e.g. 0,1,2,3",
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(conf, device)

import logging
import os
import torch
import wandb
from tqdm import tqdm


def get_logger(logfile: str):
    open(logfile, "w").close()    # clear log file
    logging.basicConfig(filename=logfile,
                        filemode="a",
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        datefmt="%d-%b-%y %H:%M:%S",
                        level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    return logger


class Trainer:
    def __init__(
        self,
        config,
        model,
        optimizer,
        scheduler,
        resume: str = "auto",
        checkpoint_name: str = "model.pt"
    ):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_name = checkpoint_name
        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.loss_fn = torch.nn.MSELoss()
        self.logger = get_logger(config.LOGFILE)

        # create folders/files
        os.makedirs(self.config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(self.config.LOGFILE), exist_ok=True)
        os.system(f"touch {self.config.LOGFILE}")

        # Init WandB
        wandb_id = None if resume == "never" else config.WANDB_ID
        wandb.init(
            project=self.config.WANDB_PROJECT,
            dir=self.config.WANDB_DIR,
            resume=resume,
            id=wandb_id,
        )

    def fit(self, train_dataloader, val_dataloader):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def mean_absolute_error(self, y_pred, y_true):
        return torch.mean(torch.abs(y_true - y_pred))

    def save_model(self):
        torch.save(self.model,
                   os.path.join(self.config.CHECKPOINT_DIR, self.checkpoint_name))

    def log(self, msg: str):
        # tqdm.write(msg)
        self.logger.debug(msg)


class MLPTrainer(Trainer):

    def fit(self, train_dataloader, val_dataloader):
        # Track gradients, weights, and biases
        wandb.watch(self.model,
                    criterion=self.loss_fn,
                    log="all",
                    log_freq=self.config.LOG_INTERVAL,
                    log_graph=True)

        plaetau_count = 0
        best_loss = 1e6
        for epoch in tqdm(range(self.config.EPOCHS), desc="Training"):
            self.log(f"Epoch {epoch + 1}")
            avg_train_loss = self.train_one_epoch(train_dataloader)
            avg_val_loss, avg_val_mae = self.evaluate(val_dataloader)

            # Logging
            self.log(f"LR: {self.scheduler.get_last_lr()[0]:.4f}")
            self.log(f"Train Loss: {avg_train_loss:.4f}")
            self.log(f"Val Loss: {avg_val_loss:.4f}")
            self.log(f"Val MAE: {avg_val_mae:.4f}")
            self.log("=========================")
            wandb.log({"train_loss": avg_train_loss,
                       "val_loss": avg_val_loss,
                       "val_mae": avg_val_mae,
                       "lr": self.scheduler.get_last_lr()[0]})

            # Update scheduler
            self.scheduler.step()

            # Save model and early stopping (if needed)
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                plaetau_count = 0
                self.save_model()
            else:
                plaetau_count += 1
                if self.config.PATIENCE > 0 and plaetau_count > self.config.PATIENCE:
                    self.log(f"Early stopping! Model has stopped improving at loss {best_loss:.4f}")
                    break

    def evaluate(self, val_dataloader):
        self.model.eval()
        torch.set_grad_enabled(False)
        running_loss = 0.0
        maes = 0.0
        for idx, (X, y) in enumerate(val_dataloader):
            X, y = X.to(self.device), y.to(self.device)
            y_pred = self.model(X, self.config.ENERGY_STD,
                                self.config.ENERGY_MEAN)
            loss = self.loss_fn(y_pred, y)
            running_loss += loss.item()
            maes += self.mean_absolute_error(y_pred, y)

        avg_loss = running_loss / len(val_dataloader)
        avg_mae = maes / len(val_dataloader)
        return avg_loss, avg_mae

    def train_one_epoch(self, train_dataloader):
        running_loss = 0.0
        last_loss = 0.0
        self.model.train()
        torch.set_grad_enabled(True)

        for idx, (X, y) in enumerate(train_dataloader):
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            y_pred = self.model(X, self.config.ENERGY_STD,
                                self.config.ENERGY_MEAN)
            loss = self.loss_fn(y_pred, y)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if (idx + 1) % self.config.LOG_INTERVAL == 0:
                last_loss = running_loss / self.config.LOG_INTERVAL
                self.log(f"\tBatch {idx + 1} - Loss: {last_loss:.4f}")
                wandb.log({"train_running_loss": last_loss})
                running_loss = 0.0

        return last_loss

    def save_model(self):
        torch.save(self.model.state_dict(),
                   os.path.join(self.config.CHECKPOINT_DIR, self.checkpoint_name))
        self.log("Model saved")


class GraphTrainer(Trainer):

    def fit(self, train_dataloader, val_dataloader):
        # Track gradients, weights, and biases
        wandb.watch(self.model,
                    criterion=self.loss_fn,
                    log="all",
                    log_freq=self.config.LOG_INTERVAL,
                    log_graph=True)

        plaetau_count = 0
        best_loss = 1e6
        for epoch in tqdm(range(self.config.EPOCHS), desc="Training"):
            self.log(f"Epoch {epoch + 1}")
            avg_train_loss = self.train_one_epoch(train_dataloader)
            avg_val_loss, avg_val_mae = self.evaluate(val_dataloader)

            # Logging
            # self.log(f"LR: {self.scheduler.get_last_lr()[0]:.4f}")
            self.log(f"Train Loss: {avg_train_loss:.4f}")
            self.log(f"Val Loss: {avg_val_loss:.4f}")
            self.log(f"Val MAE: {avg_val_mae:.4f}")
            self.log("=========================")
            wandb.log({"train_loss": avg_train_loss,
                       "val_loss": avg_val_loss,
                       "val_mae": avg_val_mae})
                    #    "lr": self.scheduler.get_last_lr()[0]})

            # Update scheduler
            # self.scheduler.step()

            # Save model and early stopping (if needed)
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                plaetau_count = 0
                self.save_model()
            else:
                plaetau_count += 1
                if self.config.PATIENCE > 0 and plaetau_count > self.config.PATIENCE:
                    self.log(f"Early stopping! Model has stopped improving at loss {best_loss:.4f}")
                    break

    def evaluate(self, val_dataloader):
        self.model.eval()
        torch.set_grad_enabled(False)
        running_loss = 0.0
        maes = 0.0
        for idx, data in enumerate(val_dataloader):
            labels = data.y.to(self.device)
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
            x = x.to(self.device)
            edge_index = edge_index.to(self.device)
            edge_attr = edge_attr.to(self.device)
            batch = batch.to(self.device)

            y_pred = self.model(x, edge_index, edge_attr, batch)
            loss = self.loss_fn(y_pred, labels)
            running_loss += loss.item()
            maes += self.mean_absolute_error(y_pred, labels)

        avg_loss = running_loss / len(val_dataloader)
        avg_mae = maes / len(val_dataloader)
        return avg_loss, avg_mae

    def train_one_epoch(self, train_dataloader):
        running_loss = 0.0
        last_loss = 0.0
        self.model.train()
        torch.set_grad_enabled(True)

        for idx, data in enumerate(train_dataloader):
            self.optimizer.zero_grad()

            # Move data to training device
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
            x = x.to(self.device)
            edge_index = edge_index.to(self.device)
            edge_attr = edge_attr.to(self.device)
            batch = batch.to(self.device)
            labels = data.y.to(self.device)

            y_pred = self.model(x, edge_index, edge_attr, batch)
            loss = self.loss_fn(y_pred, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if (idx + 1) % self.config.LOG_INTERVAL == 0:
                last_loss = running_loss / self.config.LOG_INTERVAL
                self.log(f"\tBatch {idx + 1} - Loss: {last_loss:.4f}")
                wandb.log({"train_running_loss": last_loss})
                running_loss = 0.0

        return last_loss

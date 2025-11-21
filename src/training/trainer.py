import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from ..models.two_tower_model import TwoTowerModel
from ..utils.metrics import calculate_metrics

logger = logging.getLogger(__name__)


class TwoTowerTrainer:
    """Trainer class for Two Tower Model (Recall Version)"""

    def __init__(
        self,
        model: TwoTowerModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []

    # -----------------------------
    # Optimizer / Scheduler
    # -----------------------------
    def _setup_optimizer(self):
        if self.config["optimizer"] == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"],
            )
        elif self.config["optimizer"] == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config["learning_rate"],
                momentum=self.config.get("momentum", 0.9),
                weight_decay=self.config["weight_decay"],
            )
        else:
            raise ValueError(f"Unsupported optimizer {self.config['optimizer']}")

    def _setup_scheduler(self):
        if self.config.get("scheduler") == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get("step_size", 10),
                gamma=self.config.get("gamma", 0.1),
            )
        elif self.config.get("scheduler") == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config["epochs"]
            )
        return None

    # -----------------------------
    # Train one epoch
    # -----------------------------
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc="Training")

        for batch in pbar:
            batch = self._move_batch_to_device(batch)

            user_emb, item_emb = self.model(
                batch["user_features"],
                batch["item_features"],
                batch["text_features"],
            )

            loss = self._compute_loss(user_emb, item_emb, batch["labels"])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(self.train_loader)

    # -----------------------------
    # Validate one epoch
    # -----------------------------
    def validate_epoch(self):
        self.model.eval()
        total_loss = 0.0
        preds, targs = [], []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                batch = self._move_batch_to_device(batch)
                user_emb, item_emb = self.model(
                    batch["user_features"],
                    batch["item_features"],
                    batch["text_features"],
                )

                logits = self.model.compute_similarity_dot(user_emb, item_emb)
                labels = batch["labels"].float()

                loss = F.binary_cross_entropy_with_logits(logits, labels)

                total_loss += loss.item()
                preds.extend(logits.cpu().numpy())
                targs.extend(labels.cpu().numpy())

        try:
            metrics = calculate_metrics(np.array(targs), np.array(preds))
        except:
            metrics = {}

        return total_loss / len(self.val_loader), metrics

    # -----------------------------
    # Move batch to device
    # -----------------------------
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        result = {
            "user_features": {
                k: v.to(self.device) for k, v in batch["user_features"].items()
            },
            "item_features": {
                k: v.to(self.device) for k, v in batch["item_features"].items()
            },
            "text_features": batch["text_features"].to(self.device),
            "labels": batch["labels"].to(self.device),
            "ratings": batch["ratings"].to(self.device),
        }
        return result

    # -----------------------------
    # Main training loop
    # -----------------------------
    def train(self) -> Dict:
        logger.info("====== Two-Tower Training Started ======")
        print("====== Two-Tower Training Started ======")

        best_val = float("inf")
        best_model = None

        for epoch in range(self.config["epochs"]):
            logger.info(f"Epoch {epoch+1}/{self.config['epochs']}")
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")

            train_loss = self.train_epoch()
            val_loss, val_metrics = self.validate_epoch()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)

            msg = f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}"
            print(msg)
            logger.info(msg)

            msg2 = f"Val Metrics={val_metrics}"
            print(msg2)
            logger.info(msg2)

            if val_loss < best_val:
                best_val = val_loss
                best_model = self.model.state_dict().copy()
                logger.info("— Best model updated —")
                print("— Best model updated —")

            if self.scheduler:
                self.scheduler.step()

        # Load best model
        if best_model:
            self.model.load_state_dict(best_model)
            logger.info("Loaded best model.")

        # Save training curve
        self._plot_training_history()

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_metrics": self.val_metrics,
            "best_val_loss": best_val,
        }

    # -----------------------------
    # Plot training history
    # -----------------------------
    def _plot_training_history(self):
        plot_dir = Path("outputs/results/plots")
        plot_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss curve
        axes[0].plot(self.train_losses, label="Train")
        axes[0].plot(self.val_losses, label="Validation")
        axes[0].set_title("Loss Curve")
        axes[0].legend()

        # Metric curve
        if self.val_metrics:
            for key in self.val_metrics[0].keys():
                axes[1].plot([m[key] for m in self.val_metrics], label=key)
            axes[1].set_title("Validation Metrics")
            axes[1].legend()

        plt.tight_layout()
        plt.savefig(plot_dir / "training_history.png")
        plt.close()

    # -----------------------------
    # Loss computation
    # -----------------------------
    def _compute_loss(self, user_emb, item_emb, labels):
        logits = self.model.compute_similarity_dot(user_emb, item_emb)
        return F.binary_cross_entropy_with_logits(logits, labels)


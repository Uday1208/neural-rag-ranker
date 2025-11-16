# src/training/trainer_biencoder.py
"""
Training loop implementation for the bi-encoder ranker.

This module defines a BiEncoderTrainer class that encapsulates the
core training loop, evaluation, and checkpointing logic for the
bi-encoder model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from src.training.losses import contrastive_loss
from src.utils.timing import time_block


class BiEncoderTrainer:
    """
    Trainer class for bi-encoder neural ranker models.

    This trainer handles epoch-level training, optional evaluation,
    and saving the best-performing model checkpoint.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: torch.device,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader],
        output_dir: str,
        num_epochs: int,
        temperature: float = 0.05,
        max_grad_norm: float = 1.0,
        log_every_steps: int = 100,
        save_best: bool = True,
    ) -> None:
        """
        Initialize the BiEncoderTrainer with training components.

        Args:
            model:
                Bi-encoder model to be trained.
            optimizer:
                Optimizer used for parameter updates.
            scheduler:
                Learning rate scheduler stepped after each batch.
            device:
                torch.device indicating CPU or CUDA usage.
            train_dataloader:
                DataLoader providing training batches.
            eval_dataloader:
                Optional DataLoader providing evaluation batches.
            output_dir:
                Directory path where checkpoints will be saved.
            num_epochs:
                Number of training epochs to run.
            temperature:
                Temperature used in contrastive loss.
            max_grad_norm:
                Maximum gradient norm used for clipping.
            log_every_steps:
                Interval (in steps) at which to log training loss.
            save_best:
                Whether to save the best model based on evaluation loss.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.num_epochs = num_epochs
        self.temperature = temperature
        self.max_grad_norm = max_grad_norm
        self.log_every_steps = log_every_steps
        self.save_best = save_best

        self.best_eval_loss: Optional[float] = None

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move all tensor fields in a batch dictionary onto the target device.

        Args:
            batch:
                Dictionary of tensors returned from the DataLoader.

        Returns:
            New dictionary with tensors moved onto self.device.
        """
        return {
            k: v.to(self.device) if hasattr(v, "to") else v
            for k, v in batch.items()
        }

    def _run_epoch(self, epoch_idx: int) -> float:
        """
        Run a single training epoch over the training DataLoader.

        Args:
            epoch_idx:
                Zero-based index of the current epoch.

        Returns:
            Average training loss across the epoch.
        """
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        with time_block(f"train epoch {epoch_idx + 1}"):
            for step, batch in enumerate(self.train_dataloader, start=1):
                batch = self._move_batch_to_device(batch)

                query_emb, passage_emb = self.model(
                    query_input_ids=batch["query_input_ids"],
                    query_attention_mask=batch["query_attention_mask"],
                    passage_input_ids=batch["passage_input_ids"],
                    passage_attention_mask=batch["passage_attention_mask"],
                )

                loss, _ = contrastive_loss(
                    query_emb, passage_emb, temperature=self.temperature
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                running_loss += float(loss.item())
                num_batches += 1

                if step % self.log_every_steps == 0:
                    avg_loss = running_loss / float(num_batches)
                    print(
                        f"[train] epoch={epoch_idx + 1}, step={step}, "
                        f"avg_loss={avg_loss:.4f}"
                    )

        avg_epoch_loss = running_loss / max(num_batches, 1)
        print(f"[train] epoch {epoch_idx + 1} average loss: {avg_epoch_loss:.4f}")
        return avg_epoch_loss

    def _evaluate(self, epoch_idx: int) -> Optional[float]:
        """
        Run evaluation over the eval DataLoader if available.

        Args:
            epoch_idx:
                Zero-based index of the current epoch.

        Returns:
            Average evaluation loss, or None if no eval_dataloader is set.
        """
        if self.eval_dataloader is None:
            return None

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad(), time_block(f"eval epoch {epoch_idx + 1}"):
            for batch in self.eval_dataloader:
                batch = self._move_batch_to_device(batch)

                query_emb, passage_emb = self.model(
                    query_input_ids=batch["query_input_ids"],
                    query_attention_mask=batch["query_attention_mask"],
                    passage_input_ids=batch["passage_input_ids"],
                    passage_attention_mask=batch["passage_attention_mask"],
                )

                loss, _ = contrastive_loss(
                    query_emb, passage_emb, temperature=self.temperature
                )
                total_loss += float(loss.item())
                num_batches += 1

        avg_eval_loss = total_loss / max(num_batches, 1)
        print(f"[eval] epoch {epoch_idx + 1} average loss: {avg_eval_loss:.4f}")
        return avg_eval_loss

    def _save_checkpoint(self, filename: str) -> None:
        """
        Save the current model state_dict to a checkpoint file.

        Args:
            filename:
                Name of the checkpoint file to be written.
        """
        path = self.output_dir / filename
        torch.save(self.model.state_dict(), path)
        print(f"[checkpoint] Saved model to: {path.as_posix()}")

    def train(self) -> None:
        """
        Execute the full training loop for the configured number of epochs.

        This method iterates over epochs, runs training and evaluation,
        and optionally saves the best-performing checkpoint.
        """
        for epoch_idx in range(self.num_epochs):
            train_loss = self._run_epoch(epoch_idx)
            eval_loss = self._evaluate(epoch_idx)

            if self.save_best and eval_loss is not None:
                if self.best_eval_loss is None or eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self._save_checkpoint("biencoder_best.pt")

            # Optionally save last epoch checkpoint
            if epoch_idx == self.num_epochs - 1:
                self._save_checkpoint("biencoder_last.pt")

"""Neural River Operations training callback module.

This module provides MLflow integration for model training, including metric logging,
model checkpointing, and experiment tracking.
"""

from typing import Dict, Any, Optional

import mlflow
import mlflow.pytorch
from mlflow.models import ModelSignature
from neuralhydrology.training.basetrainer import BaseTrainer
from neuralhydrology.utils.config import Config


# Constants
BEST_MODEL_METRIC = "avg_loss"


class MLflowTrainingCallback:
    """MLflow training callback for logging training progress and model checkpoints.

    This class handles the integration between the training process and MLflow,
    including metric logging, model checkpointing, and experiment tracking.

    Attributes:
        trainer: The base trainer instance
        best_val_loss: Best validation loss achieved so far
        best_model_info: Information about the best model checkpoint
        model_name: Name of the MLflow model
    """

    def __init__(self, model_name: str, signature: ModelSignature):
        """Initialize the MLflow training callback.

        Args:
            model_name: Name of the MLflow experiment
        """
        self.trainer: Optional[BaseTrainer] = None
        self.best_val_loss: float = float("inf")
        self.best_model_info: Optional[Dict[str, Any]] = None
        self.signature = signature
        self.model_name = model_name

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        """Log metrics to MLflow.

        Args:
            metrics: Dictionary of metric names and values
            step: Current training step/epoch
            prefix: Optional prefix for metric names
        """
        for name, value in metrics.items():
            mlflow.log_metric(f"{prefix}{name}", value, step=step)

    def save_best_model(self) -> None:
        """Save the best model and related artifacts to MLflow.

        This method saves:
        - Model configuration (config.yml)
        - Best PyTorch model using mlflow.pytorch.log_model
        - Training data scaler (train_data_scaler.yml)
        """
        # Save configuration
        mlflow.log_artifact(self.trainer.cfg.run_dir / "config.yml", "runs")

        # Save the best PyTorch model using mlflow.pytorch.log_model
        if self.best_model_info:
            mlflow.pytorch.log_model(
                self.trainer.model,
                artifact_path=f"runs/model_epoch{self.best_model_info['best_epoch']:03d}",
                registered_model_name=self.model_name,
                signature=self.signature,
            )

        # Save training data scaler
        mlflow.log_artifacts(self.trainer.cfg.train_dir, "runs/train_data")

    def train_and_validate(self, trainer: BaseTrainer) -> None:
        """Execute the training and validation process.

        This method handles the main training loop, including:
        - Learning rate updates
        - Training epochs
        - Validation
        - Best model tracking
        """
        self.trainer = trainer

        try:
            for epoch in range(trainer._epoch + 1, trainer._epoch + trainer.cfg.epochs + 1):
                # Update learning rate if scheduled
                if epoch in trainer.cfg.learning_rate:
                    lr = trainer.cfg.learning_rate[epoch]
                    mlflow.log_metric("learning_rate", lr, step=epoch)
                    for param_group in trainer.optimizer.param_groups:
                        param_group["lr"] = lr

                # Training phase
                trainer._train_epoch(epoch=epoch)
                train_metrics = trainer.experiment_logger.summarise()
                self.log_metrics(train_metrics, epoch, prefix="train_")

                # Validation phase (consolidated condition)
                if trainer.validator and epoch % trainer.cfg.validate_every == 0:
                    trainer.validator.evaluate(
                        epoch=epoch,
                        save_results=trainer.cfg.save_validation_results,
                        save_all_output=trainer.cfg.save_all_output,
                        metrics=trainer.cfg.metrics,
                        model=trainer.model,
                        experiment_logger=trainer.experiment_logger.valid(),
                    )

                    valid_metrics = trainer.experiment_logger.summarise()
                    print("############Validation##############")
                    print(valid_metrics)
                    print("##########################")
                    self.log_metrics(valid_metrics, epoch, prefix="valid_")

                    # Update best model if validation loss improved
                    val_loss = valid_metrics.get(BEST_MODEL_METRIC, float("inf"))
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.best_model_info = {
                            "best_epoch": epoch,
                            "best_val_loss": val_loss,
                            "best_metrics": valid_metrics,
                        }
                        # To run evaluate_and_visualize,
                        # if we use mlflow model instead, it is not needed.
                        trainer._save_weights_and_optimizer(epoch)

            # Cleanup after training
            if trainer.cfg.log_tensorboard:
                trainer.experiment_logger.stop_tb()

            # Save best model
            self.save_best_model()

        except Exception as e:
            raise e


def start_run(cfg: Config, model_name: str, signature: ModelSignature) -> Config:
    """Start a training run with MLflow integration.

    Args:
        cfg: Configuration object for the training run
        model_name: Name of the MLflow model

    Returns:
        Config: Updated configuration object after training

    Raises:
        ValueError: If the model head type is not supported
    """
    callback = MLflowTrainingCallback(model_name, signature)

    if cfg.head.lower() in ["regression", "gmm", "umal", "cmal", ""]:
        trainer = BaseTrainer(cfg=cfg)
    else:
        raise ValueError(f"Unknown head {cfg.head}.")

    trainer.initialize_training()
    callback.train_and_validate(trainer)

    return trainer.cfg

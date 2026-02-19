from pathlib import Path
from typing import List, Optional
import yaml

ALLOWED_MODULES = [
    "q_proj",
    "q_proj,v_proj",
    "q_proj,k_proj,v_proj,o_proj"
]

class TrainingConfig:
    """
    Central configuration for training a Whisper model with IA³ adapters.
    Supports YAML loading and CLI overrides.
    """

    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self._load_yaml(yaml_path)

    def _load_yaml(self, yaml_path):
        cfg = yaml.safe_load(open(yaml_path, "r"))

        # Paths
        self.features_dir: Path = Path(cfg["paths"]["features_dir"]).resolve()
        self.output_dir: Path = Path(cfg["paths"]["output_dir"]).resolve()

        # Whisper model
        self.whisper_model: str = cfg.get("whisper_model", "large")
        self.language: str = cfg.get("language", "french")
        self.task: str = cfg.get("task", "transcribe")

        # Training
        self.batch_size: int = cfg["training"].get("batch_size", 8)
        self.learning_rate: float = cfg["training"].get("learning_rate", 1e-4)
        self.num_epochs: int = cfg["training"].get("num_epochs", 30)
        self.test_evaluation: bool = cfg.get("test_evaluation", False)

        # IA³ configuration
        ia3_cfg = cfg.get("ia3", {})
        self.target_modules: List[str] = ia3_cfg.get("target_modules", ["q_proj", "v_proj"])

        # Callbacks
        callbacks_cfg = cfg.get("callbacks", {})
        self.early_stopping_patience: int = callbacks_cfg.get("early_stopping_patience", 0)
        self.early_stopping_threshold: float = callbacks_cfg.get("early_stopping_threshold", 0.1)

        # WandB
        wandb_cfg = cfg.get("wandb", {})
        self.wandb_key: Optional[str] = wandb_cfg.get("wandb_key")
        self.wandb_project: Optional[str] = wandb_cfg.get("project")

    def override(self, **kwargs):
        """
        Override values from CLI or other sources.
        """
        for k, v in kwargs.items():
            if v is not None and hasattr(self, k):
                setattr(self, k, v)

    def __repr__(self):
        return (
            f"<TrainingConfig model={self.whisper_model} "
            f"batch_size={self.batch_size} lr={self.learning_rate} "
            f"epochs={self.num_epochs} ia3_alpha={self.ia3_alpha}>"
        )

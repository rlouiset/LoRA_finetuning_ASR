from dataclasses import dataclass
from typing import Tuple, Any
from torch.utils.data import Dataset
from pathlib import Path
import torch


def load_datasets(cfg) -> Tuple[Dataset, Dataset]:
    """
    Load train and eval datasets from feature directories.
    Applies optional subsampling if cfg.test_evaluation=True.
    
    Args:
        cfg: TrainingConfig object containing features_dir and test_evaluation
    
    Returns:
        Tuple[train_dataset, eval_dataset]
    """
    train_dir = Path(cfg.features_dir) / "train"
    eval_dir = Path(cfg.features_dir) / "eval"

    train_dataset = WhisperFeaturesDataset(train_dir)
    eval_dataset = WhisperFeaturesDataset(eval_dir)

    if cfg.test_evaluation:
        # Use a small subset for faster debugging
        eval_dataset = torch.utils.data.Subset(eval_dataset, list(range(min(32, len(eval_dataset)))))

    return train_dataset, eval_dataset


@dataclass
class WhisperCollator:
    processor: Any
    include_filenames: bool = False
    remove_forbidden_keys: bool = False

    def __call__(self, features):
        # --- 1. Pad acoustic features ---
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt", padding=True
        )

        # --- 2. Pad tokenized labels ---
        labels = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            labels, return_tensors="pt", padding=True
        )

        # --- 3. Mask padding tokens ---
        batch["labels"] = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"] != 1, -100
        )

        # --- 4. Optionally remove forbidden keys for Trainer ---
        if self.remove_forbidden_keys:
            for bad_key in ["input_ids", "input_values"]:
                batch.pop(bad_key, None)

        # --- 5. Optionally include filenames for inference ---
        if self.include_filenames:
            batch["filenames"] = [f.get("filename", None) for f in features]

        return batch
    
    
    
#Data Loaders of features already processed on torch tensors ( saved time during epoch)
class WhisperFeaturesDataset(Dataset):
    def __init__(self, features_dir):
        self.files = list(features_dir.glob("*.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        return {"input_features": data["input_features"], "labels": data["labels"]}

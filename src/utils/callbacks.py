# utils/callbacks.py
import os
import csv
from transformers import TrainerCallback, EarlyStoppingCallback

class CsvWERLoggerCallback(TrainerCallback):
    """
    Logs WER metric to a CSV file at each evaluation step.
    """
    def __init__(self, csv_path="wer_log.csv"):
        self.csv_path = csv_path
        # Create CSV file with header if it does not exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "wer"])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        wer_value = metrics.get("eval_wer") or metrics.get("wer") or metrics.get("eval_eval_wer")
        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([state.global_step, wer_value])
            
            


def get_callbacks(cfg):
    """
    Returns a list of callbacks for HuggingFace Trainer.

    Args:
        cfg: TrainingConfig object with optional parameters like early_stopping_patience

    Returns:
        List of callbacks [CsvWERLoggerCallback, EarlyStoppingCallback]
    """
    # CSV logger
    csv_logger = CsvWERLoggerCallback(csv_path=os.path.join(cfg.output_dir, "wer_log.csv"))

    # Early stopping
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=cfg.early_stopping_patience,
        early_stopping_threshold=cfg.early_stopping_threshold
    )

    return [csv_logger, early_stopping]



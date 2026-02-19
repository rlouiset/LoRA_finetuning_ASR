# Fine tuning of Whisper Large using IA³ adapters for dysarthric speech (Huntington's disease)

import torch
from transformers import WhisperProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
import wandb
import evaluate
import argparse
import shutil
from utils.metrics import normalized_ponctuation, make_compute_metrics
from utils.seed import set_seed
from utils.batch_processing import WhisperCollatorFast, WhisperFeaturesDataset, load_datasets
from utils.models import load_ia3_model, load_base_model, prepare_ia3_for_training
from utils.config import TrainingConfig, ALLOWED_MODULES
from utils.callbacks import get_callbacks


def train(config: TrainingConfig):
    # -------------------------
    # Processor
    # -------------------------
    processor = WhisperProcessor.from_pretrained(
        f"openai/whisper-{config.whisper_model}",
        language=config.language,
        task=config.task
    )

    # Fix pad token to avoid attention mask warnings
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    # -------------------------
    # Dataset & Collator
    # -------------------------
    train_dataset, eval_dataset = load_datasets(config)

    collator = WhisperCollatorFast(
        processor=processor,
        include_filenames=False,
        remove_forbidden_keys=True
    )

    # -------------------------
    # Model (IA³ adapters)
    # -------------------------
    model = prepare_ia3_for_training(config, processor)

    print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())

    model.print_trainable_parameters()
    model.config.use_cache = False

    # -------------------------
    # Metrics
    # -------------------------
    metric = evaluate.load("wer")
    compute_metrics_fn = make_compute_metrics(processor, metric)

    # -------------------------
    # Callbacks
    # -------------------------
    callbacks = get_callbacks(config)

    # -------------------------
    # Training arguments
    # -------------------------
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=config.learning_rate,
        fp16=True,
        dataloader_num_workers=8,  # adjust for CPU cores
        dataloader_pin_memory=True,
        predict_with_generate=True,
        report_to=["wandb"],
        metric_for_best_model="eval_wer",
        load_best_model_at_end=True,
        greater_is_better=False,
        save_total_limit=3,
        remove_unused_columns=True,
    )

    # -------------------------
    # Trainer
    # -------------------------
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks,
    )

    # -------------------------
    # First evaluation
    # -------------------------
    trainer.evaluate()

    # -------------------------
    # Training
    # -------------------------
    trainer.train()

    # -------------------------
    # Save best checkpoint
    # -------------------------
    best_ckpt = trainer.state.best_model_checkpoint
    print("Best checkpoint:", best_ckpt)

    if best_ckpt is not None:
        shutil.copytree(best_ckpt, f"best_model_ia3", dirs_exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Whisper Large with IA³ adapters")
    parser.add_argument("--config", type=str, required=True, help="config yaml")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--ia3_alpha", type=float, default=1, help="IA³ scaling factor")
    parser.add_argument("--test_evaluation", type=bool, default=None, help="Quick eval on small subset")

    args = parser.parse_args()
    print("main")

    # -------------------------
    # YAML loading
    # -------------------------
    cfg = TrainingConfig(args.config)
    cfg.override(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        ia3_alpha=args.ia3_alpha,  # reused for IA³
        test_evaluation=args.test_evaluation,
    )

    # -------------------------
    # WandB
    # -------------------------
    wandb.login(key=cfg.wandb_key)
    wandb.init(project=cfg.wandb_project, config=cfg)

    # -------------------------
    # Train
    # -------------------------
    train(config=cfg)

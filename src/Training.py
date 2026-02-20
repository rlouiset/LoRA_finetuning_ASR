# Fine tuning of Whisper Large using Prompt adapters for dysarthric speech (Huntington's disease)

import torch
from transformers import WhisperProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
import wandb
import evaluate
import argparse
import shutil
from utils.metrics import normalized_ponctuation, make_compute_metrics
from utils.seed import set_seed
from utils.batch_processing import WhisperCollatorFast, WhisperFeaturesDataset, load_datasets
from utils.models import load_prompt_model, load_base_model, prepare_prompt_for_training
from utils.config import TrainingConfig, ALLOWED_MODULES
from utils.callbacks import get_callbacks
from torch.utils.data import DataLoader


def train(config: TrainingConfig):
    # -------------------------
    # Processor
    # -------------------------
    processor = WhisperProcessor.from_pretrained(
        f"openai/whisper-{config.whisper_model}",
        language=config.language,
        task=config.task
    )
    # Fix pad token properly
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    # -------------------------
    # Dataset & Collator
    # -------------------------
    train_dataset, eval_dataset = load_datasets(config)
    collator = WhisperCollatorFast(
        include_filenames=False,
        remove_forbidden_keys=True
    )

    eval_dataloader = DataLoader(eval_dataset, batch_size=config.batch_size, collate_fn=collator)

    # -------------------------
    # Model (prompt adapters)
    # -------------------------
    model = prepare_prompt_for_training(config, processor)

    # Ensure model is on the right device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Using device:", next(model.parameters()).device)
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
        eval_strategy="no",  # skip trainer.evaluate() to avoid input_ids error
        logging_strategy="epoch",
        learning_rate=config.learning_rate,
        fp16=True,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        predict_with_generate=False,  # use manual evaluation instead
        report_to=["wandb"],
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
        eval_dataset=None,  # evaluation handled manually
        data_collator=collator,
        compute_metrics=None,  # handled manually
        callbacks=callbacks
    )

    # -------------------------
    # Manual Evaluation before training
    # -------------------------
    print("Starting manual evaluation...")
    model.eval()
    all_preds = []
    all_refs = []

    with torch.no_grad():
        for batch in eval_dataloader:
            input_features = batch["input_features"].to(device)
            labels = batch["labels"]

            generated_ids = model.generate(input_features=input_features)
            preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
            refs = processor.batch_decode(labels, skip_special_tokens=True)

            all_preds.extend(preds)
            all_refs.extend(refs)

    wer_score = metric.compute(predictions=all_preds, references=all_refs)
    print("Validation WER:", wer_score)

    # -------------------------
    # Training
    # -------------------------
    trainer.train()

    # -------------------------
    # Save final model
    # -------------------------
    print("Saving final model...")
    model.save_pretrained(f"{config.output_dir}/final_model")
    processor.save_pretrained(f"{config.output_dir}/final_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Whisper Large with prompt adapters")
    parser.add_argument("--config", type=str, required=True, help="config yaml")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--test_evaluation", type=bool, default=None, help="Quick eval on small subset")

    args = parser.parse_args()
    print("Starting training script...")

    # -------------------------
    # YAML loading
    # -------------------------
    cfg = TrainingConfig(args.config)
    cfg.override(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
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
# Fine tuning of whisper Large using LoRA for dysarthric speech, specifically on Huntington's disease

import torch
from transformers import WhisperProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
import wandb
import evaluate
import argparse
import shutil
from utils.metrics import normalized_ponctuation,make_compute_metrics
from utils.seed import set_seed
from utils.batch_processing import WhisperCollator, WhisperFeaturesDataset,load_datasets
from utils.models import load_lora_model, load_base_model, prepare_lora_for_training
from utils.config import TrainingConfig, ALLOWED_MODULES
from utils.callbacks import get_callbacks


            
def train(config : TrainingConfig):

    
    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{config.whisper_model}",language=config.language,task=config.task)
    
    # Dataset & collator
    train_dataset, eval_dataset=load_datasets(config)
    collator = WhisperCollator(processor, include_filenames=False, remove_forbidden_keys=True)

    # Model 
    #Adapt the model with Lora
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = prepare_lora_for_training(config, processor)
    model.to(device)

    model.print_trainable_parameters()  # Print trainable parameters
    model.config.use_cache=False
    
    
    # Metrics
    metric=evaluate.load("wer")
    compute_metrics_fn=make_compute_metrics(processor, metric)
    
    #Callbacks
    callbacks=get_callbacks(config)
        
    
    # Training argument
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
        predict_with_generate=True,
        report_to=["wandb"], # Active logging WER automatiquement
        metric_for_best_model="eval_wer",
        load_best_model_at_end=True,
        greater_is_better=False,
        save_total_limit=3,
        remove_unused_columns=True,
    )
    
    
    #Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks,
    )

    #First evaluation
    trainer.evaluate()
    # Training
    trainer.train()
    
    best_ckpt = trainer.state.best_model_checkpoint
    print("Best checkpoint:", best_ckpt)

    # Copy it to a clean directory
    if best_ckpt is not None:
        shutil.copytree(best_ckpt, f"best_model_lora_rank_{config.lora_r}", dirs_exist_ok=True)
        
    


if __name__ == "__main__":
     
    
    parser=argparse.ArgumentParser(description="Fine-tune Whisper Large with LoRA")
    parser.add_argument("--config", type=str, required=True, help="config yaml")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=None,help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--lora_rank", type=int, default=None, help="LoRA rank")
    parser.add_argument("--test_evaluation",type=bool, default=None, help="Evaluate on a small subset of the eval set for quick testing")

    
    
    args =parser.parse_args()
    
    #YAML loading
    cfg=TrainingConfig(args.config)
    cfg.override(
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    num_epochs=args.num_epochs,
    lora_r=args.lora_rank,
    test_evaluation=args.test_evaluation,
)
   
    
    #Log into Wandb
    wandb.login(key=cfg.wandb_key)    
    wandb.init(project=cfg.wandb_project, config=cfg)
    
    
    train(config=cfg)


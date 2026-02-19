# ----------------------------------------------------
# Main script
# ----------------------------------------------------
# Loads model configuration, runs inference on precomputed Whisper features,
# evaluates WER for both the base and LoRA-fine-tuned models, and exports a
# detailed comparison table for subsequent analysis.

from torch.utils.data import DataLoader
from pathlib import Path
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from jiwer import wer
import yaml
import evaluate
from tqdm import tqdm 
import numpy as np
from utils.metrics import normalized_ponctuation
from utils.io import export_csv_full
from utils.batch_processing import WhisperCollator, WhisperFeaturesDataset
from utils.models import load_lora_model, load_base_model







def run_inference(model, processor, dataloader, device):
    """
    Run autoregressive inference over a dataset of precomputed features.

    Steps:
        (1) Feed log-Mel features into Whisper
        (2) Generate token sequences
        (3) Replace masked labels (-100) with pad tokens for decoding
        (4) Decode predictions and references
        (5) Normalize texts and accumulate results

    Returns:
        preds      : list of model hypotheses (decoded strings)
        refs       : list of ground-truth transcripts
        filenames  : list of corresponding sample identifiers
    """
    model.eval()
    preds, refs, filenames = [], [],[]

    for batch in tqdm(dataloader, desc="Inference"):
        feats = batch["input_features"].to(device)
        batch_filenames=batch["filenames"]

        # Génération
        generated_ids = model.generate(feats)

        # Remplacer -100 dans labels pour décodage
        labels = batch["labels"].numpy()
        labels = np.where(labels != -100,
                          labels,
                          processor.tokenizer.pad_token_id)

        pred_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        ref_texts = processor.batch_decode(labels, skip_special_tokens=True)

        pred_texts = [normalized_ponctuation(t) for t in pred_texts]
        ref_texts = [normalized_ponctuation(t) for t in ref_texts]

        preds.extend(pred_texts)
        refs.extend(ref_texts)
        filenames.extend(batch_filenames)

    return preds, refs, filenames







if __name__ == "__main__":
    
    # ---------------------------
    # Load configuration from YAML
    # ---------------------------

    with open("../configs/config_inference.yaml", "r") as f:
        config = yaml.safe_load(f)

    #------------------
    # Config extract
    #------------------
    whisper_model=config['whisper']['model']
    language=config['whisper']['language']
    task=config['whisper']['task']
    FEATURES_DIR = Path(config["paths"]["features_dir"])
    FEATURES_DIR.mkdir(exist_ok=True, parents=True)
    LORA_PATH=config["paths"]["best_model"]
    
    # ---------------------------
    # Load Whisper processor
    # ---------------------------

    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{whisper_model}", language=language, task=task)
    
    #-----Dataset ---
    
    test_dataset = WhisperFeaturesDataset(FEATURES_DIR/"test")
    
    collator = WhisperCollator(processor, include_filenames=True, remove_forbidden_keys=False)

    dataloader= DataLoader(test_dataset, batch_size=8, collate_fn=collator)
    
    # ---Device ---
    device="cuda" if torch.cuda.is_available() else "cpu"
    
    #--- charger METRIC
    wer_metric=evaluate.load("wer")
    
    
    #--------------
    #Modèle LoRA
    #--------------
    
    # Le checkpoint LoRA 
    
    
    lora_model=load_lora_model(whisper_model, processor, lora_path=LORA_PATH)
    lora_model.to(device)

    lora_preds, lora_refs,_ = run_inference(lora_model, processor, dataloader, device)
    lora_wer = 100 * wer_metric.compute(predictions=lora_preds, references=lora_refs)

    print(f"\n🟣 LoRA model WER: {lora_wer:.2f}%")
    
    
    
    #-------------------------
    #Modèle de base (sans Lora)
    #--------------------------
    print("\n Loading Base model...")
    base_model=load_base_model(whisper_model,processor)
    base_model.to(device)
    base_preds, base_refs, filenames = run_inference(base_model, processor, dataloader, device)
    base_wer=100*wer_metric.compute(predictions=base_preds, references=base_refs)
    
    
    print(f"\n🔵 Base model WER: {base_wer:.2f}%")
    

    #export WER per files
    export_csv_full(
    filenames,
    base_preds, base_refs, base_wers,
    lora_preds, lora_wers,
    path="wer_comparison.csv"
)

    # ----------------------------------------------------
    # Résumé final
    # ----------------------------------------------------
    print("\n==================== RESULTS ====================")
    print(f"🔵 Base Whisper-small WER : {base_wer:.2f}%")
    print(f"🟣 LoRA Whisper-small WER : {lora_wer:.2f}%")
    print(f"➡️ Gain : {base_wer - lora_wer:.2f} points WER")

import csv
from jiwer import wer


def export_csv_full(filenames, base_preds, base_refs,
                    lora_preds,
                    path="wer_comparison.csv"):
    """
    Export a detailed comparison table between the base model and the LoRA model.

    Each row reports:
        - filename
        - base model prediction and WER
        - LoRA model prediction and WER

    This CSV is intended for analysis, related to stage of the disease
    """
    
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index",
            "filename",
            "base_prediction",
            "reference",
            "base_WER",
            "lora_prediction",
            "lora_WER"
        ])

        for i, (fn, bp, br,lp) in enumerate(
            zip(filenames, base_preds, base_refs, lora_preds)
        ):
            bw=wer(br,bp)
            lw=wer(br,lp)
            writer.writerow([i, fn, bp, br, bw, lp, lw])

    print(f"CSV créé : {path}")
    
def compute_wer_per_sample(preds, refs):
    """
    Compute WER at the utterance level (per sample).

    Returns a list of individual WER scores, enabling detailed
    error analysis and per-file reporting.
    """
    return [wer(p, r) for p, r in zip(preds, refs)]
# Script for precomputing Whisper features (Mel spectrograms and tokenized labels)
# -------------------------------------------------------------------------------
# This script extracts Whisper-compatible acoustic features and target label tensors
# for each annotated audio segment prior to model training. The feature extraction
# is performed once, allowing subsequent training runs to load preprocessed data
# efficiently without recomputing spectrograms or tokenized transcripts.
#
# For each audio segment, we:
#   (1) load the corresponding raw waveform (with caching to avoid repeated I/O),
#   (2) resample and normalize the signal to Whisper’s expected format (16 kHz mono),
#   (3) isolate the time-defined segment boundaries,
#   (4) compute Whisper log-Mel filterbank features using the official processor,
#   (5) tokenize the associated transcript,
#   (6) save both tensors to disk for downstream training.
#
# The resulting .pt files contain:
#   { "input_features": <Tensor>[80 × T],    # Whisper filterbank features
#     "labels": <Tensor>[L] }               # tokenized transcript
#



import os
import json
import torch
from pathlib import Path
import torchaudio
from transformers import WhisperProcessor
from tqdm import tqdm
import yaml


# ---------------------------
# Load configuration from YAML
# ---------------------------

with open("config/config_features.yaml", "r") as f:
    config = yaml.safe_load(f)

SCRATCH_RAW = Path(config["paths"]["scratch_raw"])
FEATURES_DIR = Path(config["paths"]["features_dir"])
FEATURES_DIR.mkdir(exist_ok=True, parents=True)

# ---------------------------
# Load Whisper processor
# ---------------------------
processor = WhisperProcessor.from_pretrained(f"openai/whisper-{config['whisper']['model']}", language=config['whisper']['language'], task=config['whisper']['task'])

# ---------------------------
# Load JSON files describing audio segments and transcripts
# ---------------------------
train_json_path = config["paths"]["train_json"]
eval_json_path = config["paths"]["eval_json"]
test_json_path = config["paths"]["test_json"]

with open(train_json_path, "r") as f:
    train_segments = json.load(f)
with open(eval_json_path, "r") as f:
    eval_segments = json.load(f)
with open(test_json_path, "r") as f:
    test_segments = json.load(f)

# ---------------------------
# Function to extract and save Whisper features for a dataset split
# ---------------------------
def save_features(segments, split_name="train"):
    """
    Precompute Whisper features for a given dataset split.
    
    For each annotated audio segment, this function:
        - loads the waveform (using caching to avoid redundant reads),
        - resamples the audio to 16 kHz,
        - converts stereo to mono when applicable,
        - extracts the time slice corresponding to the segment,
        - computes Whisper log-Mel spectrograms,
        - tokenizes the transcript,
        - stores the resulting tensors in a .pt file.

    Parameters
    ----------
    segments : list of dict
        List of annotated segments containing:
            * file_name : audio file identifier
            * start, end : time boundaries (in seconds)
            * transcript : ground-truth text
    split_name : str
        Dataset split name ("train", "eval", or "test").
    """
    split_dir = FEATURES_DIR / split_name
    split_dir.mkdir(exist_ok=True)
    
    # Cache last loaded audio to avoid redundant disk reads
    last_file_name = None
    last_waveform = None
    last_sr = None
    
    for seg in tqdm(segments, desc=f"Processing {split_name} segments"):
        file_name = seg["file_name"]
        start = seg["start"]
        end = seg["end"]
        audio_path = SCRATCH_RAW / f"{file_name}.wav"
        
        # Construct a safe feature filename
        
        safe_file_name = file_name.replace("/", "_")
        feat_file = split_dir / f"{safe_file_name}_{int(start*1000)}_{int(end*1000)}.pt"

        if feat_file.exists():
            continue  # Skip if features already computed
        
        # Load audio (from cache if possible)
        if file_name == last_file_name:
            waveform, sr = last_waveform, last_sr
        else:
            waveform, sr = torchaudio.load(audio_path)
            last_file_name = file_name
            last_waveform = waveform
            last_sr = sr
        
        # Resample audio to Whisper's expected 16 kHz
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
            sr = 16000
        
        # Convert to mono if necessary
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Extract the relevant segment
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment = waveform[:, start_sample:end_sample].squeeze(0)
        
        # Compute Whisper input features (log-Mel spectrogram)
        input_features = processor(segment, sampling_rate=sr, return_tensors="pt").input_features.squeeze(0)
        
        # Compute Whisper input features (log-Mel spectrogram)
        labels = processor.tokenizer(seg["transcript"], return_tensors="pt").input_ids.squeeze(0)
        
        # Save features and labels as PyTorch tensor
        torch.save({"input_features": input_features, "labels": labels}, feat_file)


# ---------------------------
# Compute and save features for each split
# ---------------------------
save_features(train_segments, "train")
save_features(eval_segments, "eval")
save_features(test_segments, "test")

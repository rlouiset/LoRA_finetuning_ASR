#!/bin/bash
#SBATCH --nodelist=puck6
#SBATCH --time=48:00:00
#SBATCH --export=ALL
#SBATCH --partition=gpu-p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=/home/tleludec/Transcription_whisper/Code/fine-tuning-whisper/Script/Bash/logs/%x-%j.log

module load mbrola/3.3-dev

echo "Running via sbatch on $(hostname) on $(date)"
echo "Python version: $(python --version)"

set -e

# Project root
PROJECT_ROOT="/home/rlouiset/LoRA_finetuning_ASR"

# -------------------------------
# Training
# -------------------------------
python "$PROJECT_ROOT/src/Training.py" \
    --config "$PROJECT_ROOT/configs/config_training.yaml"

echo "Computation ended: $(date)"

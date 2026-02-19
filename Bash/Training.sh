#!/bin/bash
#SBATCH --nodelist=puck6
#SBATCH --time=48:00:00
#SBATCH --export=ALL
#SBATCH --partition=erc-cristia 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --output=/home/tleludec/Transcription_whisper/Code/fine-tuning-whisper/Script/Bash/logs/%x-%j.log

module load mbrola/3.3-dev


echo "Running via sbatch on $(hostname) on $(date)"
node=$(hostname -s)
user=$(whoami)
echo $(python --version)

set -e 

# Emplacement du script
PROJECT_ROOT="/home/tleludec/Transcription_whisper/Code/fine-tuning-whisper/Script/"


# Construire les chemins absolus
LOG_DIR="$SCRIPT_DIR/logs"

RANKS=(2 4 8)

for R in "${RANKS[@]}"; do
    echo "=============================================="
    echo "      Training with LoRA rank = $R"
    echo "=============================================="

    python "$PROJECT_ROOT/src/Training.py" \
        --config "$PROJECT_ROOT/configs/config_training.yaml" \
        --lora_rank "$R"

    echo "Fin du training pour le rank $R"
done



echo "computation end :$(date)"

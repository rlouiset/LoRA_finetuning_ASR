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


# Obtenir le chemin absolu du script SLURM
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Construire les chemins absolus
LOG_DIR="$SCRIPT_DIR/logs"
CODE_DIR="/home/tleludec/Transcription_whisper/Code/fine-tuning-whisper/Script/src"



# Aller dans le dossier contenant le script Python
cd "$CODE_DIR" || { echo "Dossier $CODE_DIR introuvable"; exit 1; }


python inference.py

echo "computation end :$(date)"

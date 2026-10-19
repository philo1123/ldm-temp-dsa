#!/usr/bin/env bash

set -e

REPO="philo1123/temp-dsa-anon"
BASE_URL="https://huggingface.co/${REPO}/resolve/main"

echo "Downloading model checkpoints from ${REPO}..."

# Create directory structure
mkdir -p models/first_stage_models/kl-f4-umm
mkdir -p models/first_stage_models/kl-f4-umm-Flow-pickFrame-ds2-residual
mkdir -p models/ldm/umm_dsa_f4_cond_BERT
mkdir -p models/ldm/umm_dsa_f4_flow-ds2-residual
mkdir -p scripts/umm_phase_CNN/predictions/dino_logreg

# Download files
wget -c ${BASE_URL}/models/first_stage_models/kl-f4-umm/best.ckpt \
  -O models/first_stage_models/kl-f4-umm/best.ckpt

wget -c ${BASE_URL}/models/first_stage_models/kl-f4-umm-Flow-pickFrame-ds2-residual/best.ckpt \
  -O models/first_stage_models/kl-f4-umm-Flow-pickFrame-ds2-residual/best.ckpt

wget -c ${BASE_URL}/models/ldm/umm_dsa_f4_flow-ds2-residual/best.ckpt \
  -O models/ldm/umm_dsa_f4_flow-ds2-residual/best.ckpt

wget -c ${BASE_URL}/models/ldm/umm_dsa_f4_cond_BERT/best.ckpt \
  -O models/ldm/umm_dsa_f4_cond_BERT/best.ckpt

wget -c ${BASE_URL}/models/selector/dino_logreg/head.joblib \
  -O scripts/umm_phase_CNN/predictions/dino_logreg/head.joblib

echo "All checkpoints downloaded successfully."
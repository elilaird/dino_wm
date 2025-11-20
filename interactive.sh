#!/usr/bin/env zsh

# Example usage:
# ./interactive.sh

srun -A coreyc_coreyc_mp_jepa_0001 \
     --partition=short \
     --time=0-04:00:00 \
     --gres=gpu:1 \
     --cpus-per-task=32 \
     --mem=96G \
     --nodes=1 \
     --pty bash
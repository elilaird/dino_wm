#!/bin/bash

# Usage: ./planning_eval.sh <branch> <time> <model_name> <epochs> [additional_args...]
# Example: ./planning_eval.sh unconstrained-second-order 1:15:00 2025-12-17/17-30-10 "10 20 30 40 50" goal_H=6

if [ $# -lt 4 ]; then
    echo "Usage: $0 <branch> <time> <model_name> <epochs> [additional_args...]"
    echo "epochs should be space-separated list like '10 20 30' or '10-50:10' for range with step"
    exit 1
fi

BRANCH="$1"
TIME="$2"
MODEL_NAME="$3"
EPOCHS_STR="$4"
shift 4

# Parse epochs - support both space-separated list and range syntax
if [[ "$EPOCHS_STR" == *-* ]]; then
    # Range syntax: start-end:step or start-end
    if [[ "$EPOCHS_STR" == *:* ]]; then
        START=$(echo "$EPOCHS_STR" | cut -d'-' -f1)
        END_STEP=$(echo "$EPOCHS_STR" | cut -d'-' -f2)
        END=$(echo "$END_STEP" | cut -d':' -f1)
        STEP=$(echo "$END_STEP" | cut -d':' -f2)
        [ -z "$STEP" ] && STEP=1
    else
        START=$(echo "$EPOCHS_STR" | cut -d'-' -f1)
        END=$(echo "$EPOCHS_STR" | cut -d'-' -f2)
        STEP=1
    fi
    EPOCHS=$(seq $START $STEP $END)
else
    EPOCHS="$EPOCHS_STR"
fi

# Base environment variables
BASE_ENV="GPU=1 CPUS=16 TYPE=plan MEM=16G BRANCH=$BRANCH PARTITION=batch TIME=$TIME CONDA_ENV=world_models"

# Loop through epochs
for epoch in $EPOCHS; do
    echo "Running evaluation for epoch $epoch..."
    CMD="$BASE_ENV ./make_sbatch.sh --config-name plan_point_maze_eval.yaml model_name=$MODEL_NAME model_epoch=$epoch $@"
    # echo "Command: $CMD"
    eval $CMD
done
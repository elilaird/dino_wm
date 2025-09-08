#!/usr/bin/env zsh
echo "SLURM_NODELIST: ${SLURM_NODELIST}"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
echo "MASTER_ADDR: ${MASTER_ADDR}"
export MASTER_PORT=29500

NUM_PROCESSES=$1
PY_FILE=$2
shift 2  # Remove first 2 args, leaving only py_args

accelerate launch --num_machines $SLURM_NNODES --num_processes=$(($SLURM_NNODES * $NUM_PROCESSES)) --machine_rank $SLURM_NODEID --main_process_port $MASTER_PORT --main_process_ip $MASTER_ADDR --dynamo_backend no --multi_gpu --rdzv_backend c10d $PY_FILE "$@"
#!/usr/bin/env zsh

# Example usage:
# GPU=4 TIME=0-04:00:00 PARTITION=batch TYPE=train ./make_sbatch.sh

DATETIME=$(date +"%Y%m%d_%H%M%S")

TIME=${TIME:-2-00:00:00}
PARTITION=${PARTITION:-batch}
TYPE=${TYPE:-train} # jupyter, eval, test
CONDA_ENV=${CONDA_ENV:-dino_clean}


GPU=${GPU:-8}
CPUS=${CPUS:-128}
MEM=${MEM:-512G}
PY_ARGS="${@}"

if [ "${PARTITION}" = "short" ]; then
    TIME="0-04:00:00"
    CPUS=16
fi

ENV_DIR=${ENV_DIR:-"/projects/coreyc/coreyc_mp_jepa/graph_world_models/ejlaird/envs"}
PROJECT_DIR=${PROJECT_DIR:-"${HOME}/Projects/dino_wm"}
DATA_DIR=${DATA_DIR:-"/lustre/smuexa01/client/users/ejlaird/dino_wm_data"}
MUJOCO_DIR=/users/ejlaird/.mujoco/mujoco210/bin

if [ "${TYPE}" = "data" ]; then
    PY_FILE="minigrid_env.py generate"
elif [ "${TYPE}" = "plan" ]; then
    PY_FILE="plan.py"
elif [ "${TYPE}" = "train" ]; then
    PY_FILE="train.py --config-name train.yaml"
fi

if [ "${TYPE}" = "jupyter" ]; then
    COMMAND="jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
else
    # Use accelerate for distributed training
    if [ "${GPU}" -gt 1 ]; then
        COMMAND="HYDRA_FULL_ERROR=1 DATASET_DIR=${DATA_DIR} accelerate launch --num_machines 1 --dynamo_backend no --num_processes=${GPU} ${PY_FILE} ${PY_ARGS}"
    else
        COMMAND="HYDRA_FULL_ERROR=1  DATASET_DIR=${DATA_DIR} python ${PY_FILE} ${PY_ARGS}"
    fi
fi

LOG_FILE="output/${TYPE}/${TYPE}_%j.out"

echo "COMMAND: GPU=${GPU} CPUS=${CPUS} MEM=${MEM} PARTITION=${PARTITION} TIME=${TIME} TYPE=${TYPE} ./make_sbatch.sh ${COMMAND}"

# write sbatch script
echo "#!/usr/bin/env zsh
#SBATCH -J ${TYPE}
#SBATCH -A coreyc_coreyc_mp_jepa_0001
#SBATCH -o output/${TYPE}/${TYPE}_%j.out
#SBATCH -c ${CPUS} --mem=${MEM}     
#SBATCH --nodes=1
#SBATCH -G ${GPU}
#SBATCH --time=${TIME} 
#SBATCH --partition=${PARTITION}

module purge
module load conda
conda activate ${ENV_DIR}/${CONDA_ENV}

which python
echo $CONDA_PREFIX

echo "COMMAND: GPU=${GPU} CPUS=${CPUS} MEM=${MEM} PARTITION=${PARTITION} TYPE=${TYPE} TIME=${TIME} ./make_sbatch.sh ${COMMAND}"

export DATA_DIR=${DATA_DIR}
export LD_LIBRARY_PATH=/users/ejlaird/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
srun bash -c \"${COMMAND}\"
" > ${TYPE}_${DATETIME}.sbatch

# submit sbatch script
sbatch ${TYPE}_${DATETIME}.sbatch

sleep 0.1
rm -f ${TYPE}_${DATETIME}.sbatch
#!/usr/bin/env zsh

# Example usage:
# GPU=4 TIME=0-04:00:00 PARTITION=batch TYPE=train ./make_sbatch.sh

DATETIME=$(date +"%Y%m%d_%H%M%S")

TIME=${TIME:-2-00:00:00}
PARTITION=${PARTITION:-batch}
TYPE=${TYPE:-train} # jupyter, eval, test

if [ "${PARTITION}" = "short" ]; then
    TIME="0-04:00:00"
fi

GPU=${GPU:-1}
CPUS=${CPUS:-16}
MEM=${MEM:-16G}
# CPUS=$((GPU * 8))  # 8 CPU cores per GPU
# MEM="${MEM:-${$((GPU * 16))}G}"  # 16GB per GPU
PY_ARGS="${@}"

# Generate random port between 6000-6100
MASTER_PORT=$((6000 + RANDOM % 101))

ENV_DIR=${ENV_DIR:-"/projects/coreyc/coreyc_mp_jepa/graph_world_models/ejlaird/envs"}
PROJECT_DIR=${PROJECT_DIR:-"${HOME}/Projects/dino_wm"}
# DATA_DIR=${DATA_DIR:-"/projects/coreyc/coreyc_mp_jepa/graph_world_models/ejlaird/data"}
DATA_DIR=${DATA_DIR:-"/lustre/smuexa01/client/users/ejlaird/dino_wm_data"}

if [ "${TYPE}" = "eval" ]; then
    PY_FILE="habitat_experiments/eval.py"
elif [ "${TYPE}" = "train" ]; then
    PY_FILE="train.py"
fi

if [ "${TYPE}" = "jupyter" ]; then
    COMMAND="jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
else
    COMMAND="HYDRA_FULL_ERROR=1 MASTER_PORT=${MASTER_PORT} DATASET_DIR=${DATA_DIR} python ${PY_FILE} ${PY_ARGS}"
fi

LOG_FILE="output/${TYPE}/${TYPE}_%j.out"
echo "COMMAND: ${COMMAND}"
echo "MASTER_PORT: ${MASTER_PORT}"

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
conda activate ${ENV_DIR}/dino_wm

which python
echo $CONDA_PREFIX

echo "COMMAND: ${COMMAND}"
echo "MASTER_PORT: ${MASTER_PORT}"

export DATA_DIR=${DATA_DIR}
export MASTER_PORT=${MASTER_PORT}
srun bash -c \"${COMMAND}\"
" > ${TYPE}_${DATETIME}.sbatch

# submit sbatch script
sbatch ${TYPE}_${DATETIME}.sbatch

sleep 0.1
rm -f ${TYPE}_${DATETIME}.sbatch
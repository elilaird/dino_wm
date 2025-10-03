#!/usr/bin/env zsh

# Example usage:
# GPU=4 TIME=0-04:00:00 PARTITION=batch TYPE=train ./make_sbatch.sh

DATETIME=$(date +"%Y%m%d_%H%M%S")

TIME=${TIME:-2-00:00:00}
PARTITION=${PARTITION:-batch}
TYPE=${TYPE:-train} # jupyter, eval, test
CONDA_ENV=${CONDA_ENV:-dino_clean}
NODES=${NODES:-1}
BRANCH=${BRANCH:-main}

GPU=${GPU:-8}
CPUS=${CPUS:-64}
MEM=${MEM:-400G}
PY_ARGS="${@}"

if [ "${PARTITION}" = "short" ]; then
    TIME="0-04:00:00"
    BRANCH=local
fi

HOME_DIR=${HOME_DIR:-"/users/ejlaird/Projects/dino_wm"}
ENV_DIR=${ENV_DIR:-"/projects/coreyc/coreyc_mp_jepa/graph_world_models/ejlaird/envs"}
WORK_DIR=${WORK_DIR:-"/projects/coreyc/coreyc_mp_jepa/graph_world_models/ejlaird/dino_wm/workdirs"}
DATA_DIR=${DATA_DIR:-"/lustre/smuexa01/client/users/ejlaird/dino_wm_data"}
MUJOCO_DIR=/users/ejlaird/.mujoco/mujoco210/bin

if [ "${TYPE}" = "jupyter" ]; then
    WORK_DIR=${HOME_DIR}
fi

echo "WORK_DIR: ${WORK_DIR}"

if [ "${TYPE}" = "minigrid" ]; then
    PY_FILE="env/minigrid/minigrid_env.py generate"
elif [ "${TYPE}" = "plan" ]; then
    PY_FILE="plan.py"
elif [ "${TYPE}" = "train" ]; then
    PY_FILE="train.py"
elif [ "${TYPE}" = "memory_maze_download" ]; then
    PY_FILE="memory_maze_download.py --output_dir ${DATA_DIR}/memory_maze"
elif [ "${TYPE}" = "memory_maze_chunk" ]; then
    PY_FILE="chunk_memory_maze_data.py"
fi

if [ "${TYPE}" = "jupyter" ]; then
    COMMAND="jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
    BRANCH="local"
elif [ "${TYPE}" = "habitat" ]; then
    COMMAND="python -m habitat_sim.utils.datasets_download --username fde8b8c1eb409327 --password bf777aaf24d6340bf647eb08c3b31817 --uids ${PY_ARGS}  --data-path ${DATA_DIR}/habitat"
    BRANCH="local"
else
    # Use accelerate for distributed training
    if [ "${GPU}" -gt 1 ]; then
        # COMMAND="HYDRA_FULL_ERROR=1 DATASET_DIR=${DATA_DIR} accelerate launch --num_machines 1 --dynamo_backend no --num_processes=${GPU} ${PY_FILE} ${PY_ARGS}"
        COMMAND="HYDRA_FULL_ERROR=1 DATASET_DIR=${DATA_DIR} ./launch.sh ${GPU} ${PY_FILE} ${PY_ARGS}"
    else
        COMMAND="HYDRA_FULL_ERROR=1  DATASET_DIR=${DATA_DIR} python ${PY_FILE} ${PY_ARGS}"
    fi
fi

LOG_FILE="output/${TYPE}/${TYPE}_%j.out"

echo "COMMAND: GPU=${GPU} CPUS=${CPUS} MEM=${MEM} BRANCH ${BRANCH} PARTITION=${PARTITION} TIME=${TIME} TYPE=${TYPE} CONDA_ENV=${CONDA_ENV} ./make_sbatch.sh ${COMMAND}"

# write sbatch script
echo "#!/usr/bin/env zsh
#SBATCH -J ${TYPE}
#SBATCH -A coreyc_coreyc_mp_jepa_0001
#SBATCH -o ${HOME_DIR}/output/${TYPE}/${TYPE}_%j.out
#SBATCH --cpus-per-task=${CPUS} 
#SBATCH --mem=${MEM}     
#SBATCH --nodes=${NODES}
#SBATCH --gres=gpu:${GPU}
#SBATCH --time=${TIME} 
#SBATCH --partition=${PARTITION}
#SBATCH --tasks-per-node=1

module purge
module load conda
module load gcc/11.2.0
module load git-lfs
conda activate ${ENV_DIR}/${CONDA_ENV}

which python


if [ \"${BRANCH}\" = \"local\" ]; then
    echo Skipping clone for local testing
else
    cd ${WORK_DIR}
    mkdir -p dino_wm_\${SLURM_JOB_ID}
    cd dino_wm_\${SLURM_JOB_ID}
    echo "Current working directory: dino_wm_\${SLURM_JOB_ID}"
    git clone git@github.com:elilaird/dino_wm.git .
    git checkout ${BRANCH}  # Use your production branch

    # Print git state information
    echo \"=== GIT STATE ===\"
    echo \"Branch: \$(git branch --show-current)\"
    echo \"Commit hash: \$(git rev-parse HEAD)\"
    echo \"Commit short: \$(git rev-parse --short HEAD)\"
    echo \"Commit message: \$(git log -1 --pretty=format:'%s')\"
    echo \"Commit author: \$(git log -1 --pretty=format:'%an <%ae>')\"
    echo \"Commit date: \$(git log -1 --pretty=format:'%ad' --date=iso)\"
    echo \"===============\"
fi

echo "WORK_DIR: \$\(pwd\)"
echo "COMMAND: GPU=${GPU} CPUS=${CPUS} MEM=${MEM} PARTITION=${PARTITION} TYPE=${TYPE} TIME=${TIME} CONDA_ENV=${CONDA_ENV} ./make_sbatch.sh ${COMMAND}"

export DATA_DIR=${DATA_DIR}
export LD_LIBRARY_PATH=/users/ejlaird/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
srun --ntasks=${NODES} --distribution=block  bash -c \"${COMMAND}\"
" > ${TYPE}_${DATETIME}.sbatch

# submit sbatch script
sbatch ${TYPE}_${DATETIME}.sbatch

sleep 0.1
rm -f ${TYPE}_${DATETIME}.sbatch
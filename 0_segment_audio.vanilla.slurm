#!/usr/bin/env bash
#SBATCH --job-name=0_segment_audio.vanilla
#SBATCH --output=0_segment_audio.vanilla.out
#SBATCH --time=5:00:00
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --cluster=mpi
#SBATCH --partition=ib

module purge
module load python/3.7.0 venv/wrap
export PYRAY_ENV=wolves
workon $PYRAY_ENV

SCRIPTS_DIR=/bgfs/jkitzes/scripts

get_ip () {
    srun --nodes=1 --ntasks=1 -w $1 $(which hostname) --all-ip-addresses
}

hostnames=($(scontrol show hostname $SLURM_JOB_NODELIST))
master_node=($(get_ip ${hostnames[0]}))

echo $hostnames $master_node

OBJECT_STORE_MEMORY=$(python $SCRIPTS_DIR/object_store_memory.py)

export REDIS_PORT=6397
export REDIS_ADDRESS=${master_node}:$REDIS_PORT
export REDIS_PASSWORD=wolvesareawesome

ray start \
    --head \
    --redis-port=$REDIS_PORT \
    --redis-password=$REDIS_PASSWORD \
    --temp-dir=$SLURM_SCRATCH \
    --plasma-directory=$SLURM_SCRATCH \
    --num-cpus=1 \
    --object-store-memory=$OBJECT_STORE_MEMORY

for worker in ${hostnames[@]:1}; do
    ssh $worker \
        bash $SCRIPTS_DIR/up.sh $WORKON_HOME $PYRAY_ENV $SLURM_SCRATCH $REDIS_ADDRESS $REDIS_PASSWORD 1 $OBJECT_STORE_MEMORY &
done
sleep 30

run_on_exit () {
    for worker in ${hostnames[@]}; do
        ssh $worker \
            bash $SCRIPTS_DIR/down.sh $WORKON_HOME $PYRAY_ENV
    done
    
    ray stop
}
trap run_on_exit EXIT

OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK python 0_segment_audio.vanilla.py \
    --input_directory=/bgfs/jkitzes/bmooreii/wolves \
    --output_directory=$SLURM_SUBMIT_DIR/splits \
    --duration=5 \
    --overlap=1 \
    --annotations \
    --labels=/ihome/sam/bmooreii/projects/opensoundscape/wolves/0-examine-dataset/labels.csv \
    --ray_address=$REDIS_ADDRESS \
    --ray_password=$REDIS_PASSWORD \
    --num_nodes=$SLURM_NNODES \
    --cores_per_node=$SLURM_CPUS_PER_TASK \
    --batch_size=$SLURM_CPUS_PER_TASK

run_on_exit

crc-job-stats.py

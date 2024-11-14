#!/bin/bash
#BSUB -J REANNqeq
#BSUB -o %J.out
#BSUB -e %J.out
#BSUB -q gpu_v100
#BSUB -gpu "num=1"
#BSUB -n 5

__conda_setup="$('/share/home/dqxie/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/share/home/dqxie/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/share/home/dqxie/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/share/home/dqxie/anaconda3/bin:$PATH"
    fi
fi
conda activate reann
nvidia-smi

export OMP_NUM_THREADS=5
#path to save the code
REANN_CORE="../reann_core_chs"

#Number of processes per node to launch
NPROC_PER_NODE=1
PBS_NUM_NODES=1

#Number of process in all modes
export LOCAL_WORLD_SIZE=`expr $PBS_NUM_NODES \* $NPROC_PER_NODE`

MASTER=`/bin/hostname -s`

MPORT=`ss -tan | awk '{print $5}' | cut -d':' -f2 | \
        grep "[2-9][0-9]\{3,3\}" | sort | uniq | shuf -n 1`

#python3 -m torch.distributed.launch --nproc_per_node=6 --nnodes=1 $COMMAND > out
#python -m torch.distributed.launch --nnodes=1 $PATH > out
torchrun --nnodes=1 --nproc_per_node=1 --master_port=12383 $REANN_CORE > modelInfo

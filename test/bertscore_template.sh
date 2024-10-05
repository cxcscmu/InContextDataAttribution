#!/bin/bash
#SBATCH --mem=32GB
#SBATCH --partition=general
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=cljiao@andrew.cmu.edu
#SBATCH --exclude=babel-3-19

set -x

__conda_setup="$('/home/cljiao/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
conda activate icdata

cd /home/cljiao/InContextDataValuation/utils

python calculate_bertscore.py \
       --start $1 \
       --end $2 \
       --train_file $3 \
       --test_file $4 \
       --outdir $5

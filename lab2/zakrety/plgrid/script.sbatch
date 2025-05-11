#!/bin/bash
#SBATCH --job-name=rl_param_study
#SBATCH --output=long/logs/param_study_%A_%a.log
#SBATCH --error=long/logs/param_study_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-39%5  # 15 combinations (5 step_sizes × 8 step_nos), max 5 concurrent

# Define parameter space
step_sizes=(0.2 0.4 0.6 0.8 1.0)
step_nos=(2 4 8 16 32 64 128 256)

# Calculate current combination
step_size=${step_sizes[$(( $SLURM_ARRAY_TASK_ID % 5 ))]}
step_no=${step_nos[$(( $SLURM_ARRAY_TASK_ID / 5 ))]}

# Run the experiment
python train.py \
    --step_size $step_size \
    --step_no $step_no \
    --experiment_rate 0.05 \
    --num_episodes 30000 \
    --corner_name "corner_c" \
    --output_dir "long/results/n${step_no}_a${step_size}"
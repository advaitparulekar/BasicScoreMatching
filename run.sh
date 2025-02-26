#!/bin/bash

# Start Python processes in separate tmux sessions with different arguments

# Start first Python process
tmux new-session -d -s p1
tmux send-keys -t p1 'source /var/local/advaitp/miniconda3/etc/profile.d/conda.sh' Enter
tmux send-keys -t p1 'conda activate CDMRL' Enter
tmux send-keys -t p1 'CUDA_VISIBLE_DEVICES=1 python naive_interventions.py --style ThreeDigit --random_intervention --second_encoder --num_latents 3 --num_epochs 10 --intervention 0' Enter

# Start second Python process
tmux new-session -d -s p2
tmux send-keys -t p2 'source /var/local/advaitp/miniconda3/etc/profile.d/conda.sh' Enter
tmux send-keys -t p2 'conda activate CDMRL' Enter
tmux send-keys -t p2 'CUDA_VISIBLE_DEVICES=2 python naive_interventions.py --style ThreeDigit --random_intervention --second_encoder --num_latents 3 --num_epochs 10 --intervention 1' Enter

# Start third Python process
tmux new-session -d -s p3
tmux send-keys -t p3 'source /var/local/advaitp/miniconda3/etc/profile.d/conda.sh' Enter
tmux send-keys -t p3 'conda activate CDMRL' Enter
tmux send-keys -t p3 'CUDA_VISIBLE_DEVICES=3 python naive_interventions.py --style ThreeDigit --random_intervention --second_encoder --num_latents 3 --num_epochs 10 --intervention 2' Enter

echo "All Python processes started."
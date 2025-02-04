#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <run_name>"
    exit 1
fi

PROJECT=isaac-locomotion-sigma
ENV=Isaac-Velocity-Flat-Sigmaban-Play
TAGS=locomotion
ALGO=ppo
EXP_ID=`python next_run_id.py $ALGO $ENV`

# killall -9 python

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task $ENV-v0 \
    --experiment_name $ALGO \
    --load_run $1 \
    --num_env 1 \
    # --headless --video --video_length 2000 \


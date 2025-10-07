#!/bin/bash

# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0      # if libgpu_partition.so confilts with gym and robosuite
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
# export MUJOCO_GL=egl
# export PYOPENGL_PLATFORM=egl

source activate bcib3

EVAL_ALL=$1

# bash libero_exp/scripts/eval_libero_all.sh False
# bash libero_exp/scripts/eval_libero_all.sh True

LOAD_PATHS=(
    # the directory of the checkpoints
)

for LOAD_PATH in "${LOAD_PATHS[@]}"; do
    python eval_libero.py \
        --config-path=${LOAD_PATH} \
        --config-name=config.yaml \
        eval.load_path=${LOAD_PATH} \
        eval.eval_all=${EVAL_ALL}
done

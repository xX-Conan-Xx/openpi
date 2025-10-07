#!/bin/bash

# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0      # if libgpu_partition.so confilts with gym and robosuite
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
# export MUJOCO_GL=egl
# export PYOPENGL_PLATFORM=egl

source activate bcib

LOAD_PATH=$1
EVAL_ALL=$2

# bash libero_exp/scripts/eval_libero.sh 'outputs/libero/bc_policy/vilt/libero_goal/1130_1137_seed0' False

python eval_libero.py \
    --config-path=${LOAD_PATH} \
    --config-name=config.yaml \
    eval.load_path=${LOAD_PATH} \
    eval.eval_all=${EVAL_ALL}

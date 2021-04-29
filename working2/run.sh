#!/bin/bash

# Copyright 2021 Ibuki Kuroyanagi.
# Created by Ibuki Kuroyanagi

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

. ./path.sh || exit 1
. ./cmd.sh || exit 1

# basic setting
verbose=1               # verbosity level, higher is more logging
stage=0                 # stage to start
stop_stage=100          # stage to stop
n_gpus=4                # number of gpus for training
n_jobs=4                # number of parallel jobs in feature extraction
speed_facters="0.9 1.1" # The facter of data augmentation.
conf=conf/Cnn14_DecisionLevelAtt.yaml

# directory related
dumpdir=dump
expdir=exp # directory to save experiments
# tag for manangement of the naming of experiments
resume=""
# evaluation related
train_file="arai_train_tf_efficientnet_b7_ns_mgpu"

. ./utils/parse_options.sh || exit 1
set -euo pipefail
tag="${train_file}/lr3e_4"
if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    log "Stage 1: Network training."
    outdir=${expdir}/${tag}
    [ ! -e "${outdir}" ] && mkdir -p "${outdir}"
    log "Training start. See the progress via ${outdir}/${train_file}.log"
    if [ "${n_gpus}" -gt 1 ]; then
        train="python ../input/modules/distributed/launch.py --master_port 29504 --nproc_per_node ${n_gpus} ${train_file}.py"
    else
        train="python ${train_file}.py"
    fi
    # shellcheck disable=SC2086,SC2154
    ${train_cmd} --num_threads "${n_jobs}" --gpu "${n_gpus}" "${outdir}/${train_file}.log" \
        ${train} \
        --resume ${resume} \
        --outdir ${outdir} \
        --n_gpus ${n_gpus} \
        --verbose ${verbose}

    log "Successfully finished the training."
fi

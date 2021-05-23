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
verbose=1    # verbosity level, higher is more logging
stage=1      # stage to start
stop_stage=1 # stage to stop
n_gpus=1     # number of gpus for training
n_jobs=2     # number of parallel jobs in feature extraction
fold=4
# directory related
expdir=exp # directory to save experiments
# tag for manangement of the naming of experiments
# resume="exp/train_mobilenetv3_relabel/mixup3/best_score/best_scorefold${fold}bce.pkl"
resume=""
# evaluation related
train_file="train_mobilenetv3_relabel"
infer_file="infer_mobilenetv3_relabel"

save_name="sec5"
. ./utils/parse_options.sh || exit 1
set -euo pipefail

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    log "Stage 0: Re-labeled Network training."
    tag="${train_file}/sec5"
    outdir=${expdir}/${tag}
    [ ! -e "${outdir}" ] && mkdir -p "${outdir}"
    log "Training start. See the progress via ${outdir}/${train_file}${save_name}${fold}.log"
    if [ "${n_gpus}" -gt 1 ]; then
        train="python ../input/modules/distributed/launch.py --master_port 2952${fold} --nproc_per_node ${n_gpus} ${train_file}.py"
    else
        train="python ${train_file}.py"
    fi
    # shellcheck disable=SC2086,SC2154
    ${train_cmd} --num_threads "${n_jobs}" --gpu "${n_gpus}" "${outdir}/${train_file}${save_name}${fold}.log" \
        ${train} \
        --resume ${resume} \
        --outdir ${outdir} \
        --n_gpus ${n_gpus} \
        --fold ${fold} \
        --save_name ${save_name} \
        --verbose ${verbose}

    log "Successfully finished the training."
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    log "Stage 1: Re-labeled Network inference."
    tag="${infer_file}/sec5"
    outdir=${expdir}/${tag}
    resume=""
    for i in {0..4}; do
        resume+="exp/train_mobilenetv3_relabel/sec5/best_score/best_scorefold${i}sec5.pkl "
    done
    [ ! -e "${outdir}" ] && mkdir -p "${outdir}"
    log "Inference start. See the progress via ${outdir}/${infer_file}${save_name}.log"
    # shellcheck disable=SC2086,SC2154
    ${train_cmd} --num_threads "${n_jobs}" --gpu "1" "${outdir}/${infer_file}${save_name}.log" \
        python ${infer_file}.py \
        --resume ${resume} \
        --outdir ${outdir} \
        --save_name ${save_name} \
        --verbose ${verbose}

    log "Successfully finished the inference."
fi

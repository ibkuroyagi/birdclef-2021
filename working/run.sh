#!/bin/bash

# Copyright 2020 Ibuki Kuroyanagi.
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
n_gpus=1                # number of gpus for training
n_jobs=4                # number of parallel jobs in feature extraction
type=wave               # preprocess type.
cal_type=1              # if 1 -> statistic, else -> load cache pkl.
speed_facters="0.9 1.1" # The facter of data augmentation.
conf=conf/Cnn14_DecisionLevelAtt.yaml

# directory related
datadir="../input/rfcx-species-audio-detection"
dumpdir=dump
expdir=exp                        # directory to save experiments
tag="Cnn14_DecisionLevelAtt/base" # tag for manangement of the naming of experiments
cache_path="../input/pretrained/Cnn14_DecisionLevelAtt.pth"
resume=""
# evaluation related
checkpoint="best_score" # path of checkpoint to be used for evaluation
checkpoints=""

. utils/parse_options.sh || exit 1
set -euo pipefail

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    log "Stage 0: Feature extraction."
    statistic_path="${dumpdir}/cache/${type}.pkl"
    [ ! -e "${dumpdir}/cache" ] && mkdir -p "${dumpdir}/cache"
    log "Feature extraction. See the progress via ${dumpdir}/${type}/preprocess.log"
    # shellcheck disable=SC2086,SC2154
    ${train_cmd} --num_threads "${n_jobs}" "${dumpdir}/${type}/preprocess.log" \
        python ../input/modules/bin/preprocess.py \
        --datadir "${datadir}" \
        --dumpdir "${dumpdir}" \
        --config "${conf}" \
        --statistic_path "${statistic_path}" \
        --cal_type "${cal_type}" \
        --type "${type}" \
        --facter "1.0" \
        --verbose "${verbose}"
    log "Successfully calculate logmel spectrogram."
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    log "Stage 1: Speed augmentation."
    if [ -n "${speed_facters}" ]; then
        statistic_path="${dumpdir}/cache/${type}.pkl"
        for facter in ${speed_facters}; do
            log "Feature extraction. See the progress via ${dumpdir}/${type}_sp${facter}/preprocess.log"
            # shellcheck disable=SC2086,SC2154
            ${train_cmd} --num_threads "${n_jobs}" "${dumpdir}/${type}_sp${facter}/preprocess.log" \
                python ../input/modules/bin/preprocess.py \
                --datadir "${datadir}" \
                --dumpdir "${dumpdir}" \
                --config "${conf}" \
                --statistic_path "${statistic_path}" \
                --cal_type "0" \
                --type "${type}_sp${facter}" \
                --facter "${facter}" \
                --verbose "${verbose}"
            log "Successfully calculated logmel spectrogram(facter is ${facter})."
        done
    else
        echo "Skip Stage 1."
    fi
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    log "Stage 2: Network training."
    outdir=${expdir}/${tag}
    dumpdirs="${dumpdir}/${type} "
    if [ -n "${speed_facters}" ]; then
        for facter in ${speed_facters}; do
            dumpdirs+="${dumpdir}/${type}_sp${facter} "
        done
    fi
    log "Training start. See the progress via ${outdir}/sed_train.log"
    if [ "${n_gpus}" -gt 1 ]; then
        chmod 755 ../input/modules/bin/sed_train.py
        train="python ../input/modules/distributed/launch.py --nproc_per_node ${n_gpus} ../input/modules/bin/sed_train.py"
    else
        train="python ../input/modules/bin/sed_train.py"
    fi
    # shellcheck disable=SC2086,SC2154
    ${train_cmd} --num_threads "${n_jobs}" --gpu "${n_gpus}" "${outdir}/sed_train.log" \
        ${train} \
        --datadir "${datadir}" \
        --dumpdirs ${dumpdirs} \
        --outdir "${outdir}" \
        --cache_path "${cache_path}" \
        --config "${conf}" \
        --resume ${resume} \
        --verbose "${verbose}"

    log "Successfully finished the training."
fi

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    log "Stage 3: Network inference."
    outdir=${expdir}/${tag}/${checkpoint}
    if [ -z "${checkpoints}" ]; then
        for fold in {0..4}; do
            checkpoints+="${expdir}/${tag}/${checkpoint}/${checkpoint}fold${fold}.pkl "
        done
    fi
    dumpdirs="${dumpdir}/${type} "
    if [ -n "${speed_facters}" ]; then
        outdir+="/sp"
        for facter in ${speed_facters}; do
            dumpdirs+="${dumpdir}/${type}_sp${facter} "
            outdir+="_${facter}"
        done
    fi
    [ ! -e "${outdir}" ] && mkdir -p "${outdir}"
    log "Inference start. See the progress via ${outdir}/sed_inference.log"
    # shellcheck disable=SC2086
    ${cuda_cmd} --num_threads "${n_jobs}" --gpu "1" "${outdir}/sed_inference.log" \
        python ../input/modules/bin/sed_inference.py \
        --datadir "${datadir}" \
        --dumpdirs ${dumpdirs} \
        --outdir "${outdir}" \
        --config "${conf}" \
        --checkpoints ${checkpoints} \
        --verbose "${verbose}"
    log "Successfully finished the inference."
fi

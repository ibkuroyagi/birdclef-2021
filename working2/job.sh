#!/bin/bash

# Copyright 2021 Ibuki Kuroyanagi.
# Created by Ibuki Kuroyanagi

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

. ./path.sh || exit 1
. ./cmd.sh || exit 1

stage=1
stop_stage=1

. ./utils/parse_options.sh || exit 1
set -euo pipefail

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    echo "Start Stage 0 split_train_short"
    # shellcheck disable=SC2086,SC2154
    ${train_cmd} --num_threads 2 "split_train_short.log" \
        python split_train_short.py
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "Start Stage 1 make best threshold (magic)"
    # shellcheck disable=SC2086,SC2154
    ${train_cmd} --num_threads 2 "make_best_threshold.log" \
        python make_best_threshold.py
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "Start Stage 2 relabel"
    # shellcheck disable=SC2086,SC2154
    ${train_cmd} --num_threads 2 --gpu 1 "relabel.log" \
        python relabel.py
fi

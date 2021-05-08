# cuda related
export CUDA_HOME=/usr/local/cuda-11.1
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# path related
export PRJ_ROOT="${PWD}/.."
# if [ -e "${PRJ_ROOT}/tools/centos_venv/bin/activate" ]; then
#     # shellcheck disable=SC1090
#     . "${PRJ_ROOT}/tools/centos_venv/bin/activate"
# fi
if [ -e "${PRJ_ROOT}/tools/activate_python.sh" ]; then
    # shellcheck disable=SC1090
    . "${PRJ_ROOT}/tools/activate_python.sh"
fi
export PATH="${PATH}:${PRJ_ROOT}/utils"
# python related
export OMP_NUM_THREADS=2
export PYTHONIOENCODING=UTF-8
export MPL_BACKEND=Agg
export LC_CTYPE=en_US.UTF-8

#!/bin/bash
cd /work2/i_kuroyanagi/kaggle/BridCLEF/working2
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  set | grep SLURM | while read line; do echo "# $line"; done
  echo -n '# '; cat <<EOF
python relabel.py 
EOF
) >relabel.log
if [ "$CUDA_VISIBLE_DEVICES" == "NoDevFiles" ]; then
  ( echo CUDA_VISIBLE_DEVICES set to NoDevFiles, unsetting it... 
  )>>relabel.log
  unset CUDA_VISIBLE_DEVICES.
fi
time1=`date +"%s"`
 ( python relabel.py  ) &>>relabel.log
ret=$?
sync || truetime2=`date +"%s"`
echo '#' Accounting: begin_time=$time1 >>relabel.log
echo '#' Accounting: end_time=$time2 >>relabel.log
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>relabel.log
echo '#' Finished at `date` with status $ret >>relabel.log
[ $ret -eq 137 ] && exit 100;
touch ./q/done.31330
exit $[$ret ? 1 : 0]
## submitted with:
# sbatch --export=PATH --ntasks-per-node=1  -p centos,ubuntu -x million4,nsx,jesko,chikaku1,aventador --gres=gpu:1 --cpus-per-task 2 --ntasks-per-node=1  --open-mode=append -e ./q/relabel.log -o ./q/relabel.log  /work2/i_kuroyanagi/kaggle/BridCLEF/working2/./q/relabel.sh >>./q/relabel.log 2>&1

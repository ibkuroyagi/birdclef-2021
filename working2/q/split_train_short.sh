#!/bin/bash
cd /work2/i_kuroyanagi/kaggle/BridCLEF/working2
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  set | grep SLURM | while read line; do echo "# $line"; done
  echo -n '# '; cat <<EOF
python split_train_short.py 
EOF
) >split_train_short.log
if [ "$CUDA_VISIBLE_DEVICES" == "NoDevFiles" ]; then
  ( echo CUDA_VISIBLE_DEVICES set to NoDevFiles, unsetting it... 
  )>>split_train_short.log
  unset CUDA_VISIBLE_DEVICES.
fi
time1=`date +"%s"`
 ( python split_train_short.py  ) &>>split_train_short.log
ret=$?
sync || truetime2=`date +"%s"`
echo '#' Accounting: begin_time=$time1 >>split_train_short.log
echo '#' Accounting: end_time=$time2 >>split_train_short.log
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>split_train_short.log
echo '#' Finished at `date` with status $ret >>split_train_short.log
[ $ret -eq 137 ] && exit 100;
touch ./q/done.8824
exit $[$ret ? 1 : 0]
## submitted with:
# sbatch --export=PATH --ntasks-per-node=1  -p ubuntu -x huracan,nsx --cpus-per-task 2 --ntasks-per-node=1  --open-mode=append -e ./q/split_train_short.log -o ./q/split_train_short.log  /work2/i_kuroyanagi/kaggle/BridCLEF/working2/./q/split_train_short.sh >>./q/split_train_short.log 2>&1

#!/bin/sh
NAME=exp_`date "+%m_%d_%H_%M_%S"`
ALGO=$1
EXP_FILE=$2
source scripts/local_env_setup.sh
python -m es_distributed.main master --master_socket_path /tmp/es_redis_master.sock --algo $ALGO --exp_file $EXP_FILE
